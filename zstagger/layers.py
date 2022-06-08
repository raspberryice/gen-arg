import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout 
from torch_struct import LinearChainCRF
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

   
    def forward(self, embeds):
        lstm_out, _ = self.lstm(embeds)
        return lstm_out

# From OneIE 
def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

class MLP(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class BERTEncoder(nn.Module):
    def __init__(self, args, bert_dim):
        super(BERTEncoder, self).__init__() 
        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.bert_dropout = Dropout(p=args.bert_dropout)
        self.bert_dim = bert_dim
        self.bert_pooler = nn.Linear(bert_dim*2, bert_dim) 

    def forward(self, input_ids, attention_mask, token_lens):
        '''
        token_lens: list
        '''
        batch_size = input_ids.size(0)
        all_bert_outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask, output_hidden_states=True)
        bert_outputs = all_bert_outputs[0]
        extra_bert_outputs = all_bert_outputs[2][-3] # from OneIE 
        bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2) # (batch, bpe_max_len, 2*hidden_dim)
        bert_outputs = self.bert_pooler(bert_outputs) # (batch, bpe_max_len, hidden_dim)

        # average all pieces for multi-piece words
        assert(len(token_lens) > 0)
        idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
        idxs = input_ids.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1 # shift 1 for [CLS]
        masks = bert_outputs.new(masks).unsqueeze(-1)
        bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
        bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
        bert_outputs = bert_outputs.sum(2)

        bert_outputs = self.bert_dropout(bert_outputs)

        return bert_outputs 


class PrototypeNetworkHead(nn.Module):
    def __init__(self,configs, feature_size, n_classes, class_vectors):
        super(PrototypeNetworkHead, self).__init__() 
        self.configs = configs 
        C = n_classes
        self.projection = MLP(dimensions=[feature_size, n_classes], activation='relu', dropout_prob=0.2)
        self.transition = nn.Linear(C, C) # For CRF
        # normalize class vectors 
        class_vectors = F.normalize(class_vectors, dim=1)
        self.class_vectors = nn.Parameter(class_vectors, requires_grad=False)
        null_vec = F.normalize(torch.rand(1,feature_size), dim=1)
        self.null_vec = nn.Parameter(null_vec, requires_grad=False) # for class null 

    def forward(self, features, lengths):
        class_vectors = torch.cat([self.null_vec, self.class_vectors], dim=0) # train the class vectors as well 
        if self.configs.no_projection:
            # for the complete zero-shot setting 
            final =  torch.einsum('ijk,lk->ijl', features, class_vectors)
        else:
            projected_x = self.projection(features)
            projected_class = self.projection(class_vectors)
            final = torch.einsum('ijk,lk->ijl', projected_x, projected_class)
        # CRF
        batch, N, C = final.shape

        if self.configs.token_classification:
            return final  
        
        if self.configs.no_projection:
            vals = final.view(batch, N, C, 1)[:, 1:N]
            vals = vals.expand(batch, N-1, C, C).clone() # without transition 
        else:
            vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.weight.view(1, 1, C, C)

        
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0]
        dist = LinearChainCRF(vals, lengths=lengths)
        return dist


class CRFFeatureHead(nn.Module):
    def __init__(self, configs, feature_size, n_classes):
        super(CRFFeatureHead, self).__init__()
        self.configs = configs
        C = n_classes

        # Prepare LSTM
        if configs.use_bilstm:
            hidden_size = configs.bilstm_hidden_size
            self.bilstm = BiLSTM(feature_size, hidden_size)
            self.dropout = nn.Dropout(p=configs.bilstm_dropout)
            encoder_size = 2 * hidden_size
            print('Prepare BiLSTM')
        else:
            encoder_size = feature_size

        # Maps into output space.
        self.hidden2output = nn.Linear(encoder_size, C)

        # Parameters for CRF layer
        self.transition = nn.Linear(C, C) # For CRF
        
    def forward(self, features, lengths):
        # Bidirectional LSTM + Projection Layer
        if self.configs.use_bilstm:
            x = self.dropout(self.bilstm(features))
        else:
            x = features
        final = self.hidden2output(x)
        # CRF
        batch, N, C = final.shape

        vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.weight.view(1, 1, C, C)
        # vals = final.view(batch, N, C, 1)[:, 1:N]
        # vals = vals.expand(batch, N-1, C, C).clone() # without transition 
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0]
        dist = LinearChainCRF(vals, lengths=lengths)
        return dist


class ZeroShotCollapsedTransitionCRFHead(nn.Module):
    def __init__(self, configs, feature_size, n_classes, proj_dim, class_vectors, max_classes=None):
        super(ZeroShotCollapsedTransitionCRFHead, self).__init__()
        self.configs = configs
        C = n_classes
        if not max_classes:
            m = n_classes
        else:
            m = max_classes 
        self.C = C
        self.m = m
        self.proj_dim = proj_dim
        encoder_size = feature_size
        self.feature_size = feature_size
        class_vectors = F.normalize(class_vectors, dim=1)
        self.class_vectors = nn.Parameter(class_vectors, requires_grad=False)
        null_vec = F.normalize(torch.rand(1,feature_size), dim=1)
        self.null_vec = nn.Parameter(null_vec, requires_grad=False) # for class null, 
        # Reference vectors 
        noise_scale = 0.01 
        self.reference_vecs = nn.Parameter(torch.eye(m, encoder_size, dtype=torch.float) + torch.rand((m, encoder_size))*noise_scale)
        # Transition matrix 
        self.self_transition_diag = nn.Parameter(torch.rand(proj_dim)) # actually diagonal  
        self.null_transition_diag = nn.Parameter(torch.rand(proj_dim))
        # projection matrix 
        self.register_buffer('M',  torch.zeros((feature_size, proj_dim)))
        # load classes vectors 
        print('initializing projection M...')
        self.update_projection() 


        

    def update_projection(self):
        '''
        class_vectors: (C, encoder_size)
        m: number of reference vectors 
        d: dimension of projection

        Output:
        M: projection matrix (encoder_size, d)
        '''
        class_vectors = torch.cat([self.null_vec, self.class_vectors], dim=0).detach()  # (C, feature_size)
        ref_vec = self.reference_vecs.detach() 
        C, encoder_size = class_vectors.shape
        # TODO: compute modified reference vector 
        mod_ref_vec = (ref_vec - torch.mean(ref_vec, dim=0)) * C/(C-1)
         
        D = F.normalize(class_vectors, dim=1) - F.normalize(mod_ref_vec[:C, :], dim=1) 
        # QR decomposition of D 
        Q,R = torch.qr(D.transpose(0,1), some=False) # complete QR 
        M = Q[:, C: (C+self.proj_dim)] # take proj_dim columns from Q 
        assert(M.size(0) == self.feature_size)
        assert(M.size(1) == self.proj_dim)
        return M 


    
    def compute_transition(self, projected_ref, projected_x):
        '''
        projected_x: (batch, seq,  proj_dim)
        projected_ref: (m, proj_dim)

        self.transition_mat: (proj_dim, proj_dim)


        Output:
        transition matrix (batch, seq, m, m )
        '''

        batch, seq, proj_dim = projected_x.shape
        m = projected_ref.size(0) 
        full_transition = torch.zeros((batch, seq, m,m)).to(projected_ref.device)
        self_transition_mat = torch.diag(self.self_transition_diag)
        null_transition_mat = torch.diag(self.null_transition_diag)
        trans_ii = projected_x.reshape(-1, self.proj_dim).matmul(self_transition_mat).matmul(projected_ref.transpose(0, 1)) #(batch*seq, m)
        trans_ii = trans_ii.reshape(batch, seq, m )

        projected_null = projected_ref[0, :] # (proj_dim)
        trans_0 = projected_x.reshape(-1, self.proj_dim).matmul(null_transition_mat).matmul(projected_null.unsqueeze(1)) #(batch*seq, 1)
        # compute self-attention 
        # cross_attn = torch.einsum("ij,ij->ii", projected_ref, projected_ref) # (m, m)
        # attn_scores= F.softmax(cross_attn, dim=1) #(m,m)
        # attn_vecs = torch.einsum("ik,ij->ikj", attn_scores, projected_ref).sum(dim=1) # (m, proj_dim) 
        for i in range(m):
            full_transition[:, :, i,i] = trans_ii[:, :, i]
            full_transition[:, :, 0, i] = trans_0.reshape(batch, seq)
            full_transition[:, :, i, 0] = trans_0.reshape(batch, seq)

        return full_transition 


    def update_params(self):
        self.M = self.update_projection()
        return 

    def normalize_params(self):
        self.reference_vecs.data = F.normalize(self.reference_vecs, dim=1)
        return 

    def regularize_params(self):
        '''
        Orthonormal regularization.
        '''
        return torch.norm(self.reference_vecs.matmul(self.reference_vecs.t()) - torch.eye(self.m).to(self.reference_vecs.device), p=2) 

    def forward(self, features, lengths):
                        
        x = features # (batch, seq, encoder_size)
        projected_x = torch.matmul(x, self.M) # (batch, seq,  proj_dim)
        projected_ref = torch.matmul(self.reference_vecs[:self.C, :], self.M) # (m, proj_dim)
        
        final = torch.einsum('ijk,lk->ijl', projected_x, projected_ref) # (batch, seq, m)
        
        if self.configs.token_classification:
            return final 

        transition = self.compute_transition(projected_ref, projected_x) # (batch, seq, m, m )
        # CRF
        batch, N, C = final.shape
        
        if self.configs.use_transition:
            vals = final.view(batch, N, C, 1)[:, 1:N] + transition[:, 1:N, :, :]
        else:
            vals = final.view(batch, N, C, 1)[:, 1:N]
            vals = vals.expand(batch, N-1, C, C).clone() 
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0]
        dist = LinearChainCRF(vals, lengths=lengths)
        return dist


class ZeroShotCRFFeatureHead(nn.Module):
    def __init__(self, configs, feature_size, n_classes, proj_dim, class_vectors, max_classes=None):
        super(ZeroShotCRFFeatureHead, self).__init__()
        self.configs = configs
        C = n_classes
        if not max_classes:
            m = n_classes
        else:
            m = max_classes 
        self.C = C
        self.m = m
        self.proj_dim = proj_dim
        encoder_size = feature_size
        self.class_vectors = class_vectors 
        self.null_vec = nn.Parameter((torch.rand(encoder_size))) # for class null
        # Reference vectors 
        noise_scale = 0.01
        self.reference_vecs = nn.Parameter(torch.eye((m, encoder_size), dtype=torch.float) + torch.rand((m, encoder_size))*noise_scale)
        # Transition matrix 
        self.transition_mat = nn.Parameter(torch.rand(2*proj_dim, C*C))
        self.transition = nn.Parameter(torch.zeros((C*C)), requires_grad=False) 
        # projection matrix 
        self.M = nn.Parameter(torch.zeros((feature_size, proj_dim)), requires_grad=False) 
        # load classes vectors 

        

    def update_projection(self):
        '''
        class_vectors: (C, encoder_size)
        m: number of reference vectors 
        d: dimension of projection

        Output:
        M: projection matrix (encoder_size, d)
        '''
        class_vectors = torch.cat([self.class_vectors, self.null_vec], dim=0)
        C, encoder_size = class_vectors.shape 
        D = F.normalize(class_vectors, dim=1) - F.normalize(self.reference_vecs.weight[:C], dim=1) 
        # QR decomposition of D 
        Q,R = torch.qr(D, some=False) # complete QR 
        M = Q[:, C: (C+self.proj_dim+1)] # take proj_dim columns from Q 

        return M 


    
    def compute_transition(self, C):
        '''
        M: projection matrix (encoder_size, d)
        transition_mat: 2d*c *c 


        Output:
        transition matrix 
        '''
        projected_ref = torch.matmul(self.reference_vecs, self.M) # (m, proj_dim)
        # compute self-attention 
        cross_attn = torch.einsum("ij,ij->ii", projected_ref, projected_ref) # (m, m)
        attn_scores= F.softmax(cross_attn, dim=1) #(m,m)
        attn_vecs = torch.einsum("ik,ij->ikj", attn_scores, projected_ref).sum(dim=1) # (m, proj_dim) 
        transition = torch.zeros((C, C)).to(self.reference_vecs.device)
        for i in range(C):
            for j in range(C):
                score = torch.dot(self.transition_mat, torch.cat([attn_vecs[i,:], attn_vecs[j,:]], dim=0)) 
                transition[i,j] = score 
       
        return transition 


    def update_params(self):
        self.M = self.update_projection()
        self.transition = self.compute_transition(self.C)
        return 
    
    def forward(self, features, lengths):
                        
        x = features # (batch, seq, encoder_size)
        projected_x = torch.matmul(x, self.M) # (batch, seq,  proj_dim)
        projected_ref = torch.matmul(self.reference_vecs[:self.C, :], self.M) # (m, proj_dim)
        final = torch.einsum('ijk,lk->ijl', projected_x, projected_ref) # (batch, seq, m)
        # CRF
        batch, N, C = final.shape
        vals = final.view(batch, N, C, 1)[:, 1:N] + self.transition.view(1, 1, C, C)
        vals[:, 0, :, :] += final.view(batch, N, 1, C)[:, 0]
        dist = LinearChainCRF(vals, lengths=lengths)
        return dist
