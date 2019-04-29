import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):

    def __init__(self, dropout=0.0):

        super(Attention, self).__init__()

        self.dropout = dropout
    
    def scaled_dot(self, Q, K):
        """ 点积操作: 
        Returns: [batch, Q_len, K_len], 除以d_k 防止溢出
        """
        d_k = Q.size(-1)
        return torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    def forward(self, Q, K, V):
        """
        Args:
            Q: [batch_size, Q_len, dim]
            K: [batch_size, k_len, dim]
            V: [batch_size, K_len, dim]
        Returns:

        """
        
        # Score 的计算 
        
        scores =  self.scaled_dot(Q, K)
        
        # Attention value 的计算
        atten_weights = F.softmax(scores, dim=-1)  # 注意力权重
        atten_weights = F.dropout(atten_weights, p=self.dropout)

        atten_vals = torch.matmul(atten_weights, V)  # 最终的向量表示
        return atten_vals, atten_weights



        


