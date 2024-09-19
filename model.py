# model.py

import torch
import torch.nn as nn
import math


class FusionLayer(nn.Module):
    """
    Fusion layer to combine word embeddings, POS tag embeddings, and rule embeddings using a gating mechanism.
    """
    def __init__(self, embed_dim, pos_embed_dim, rule_embed_dim, fusion_dim):
        super(FusionLayer, self).__init__()
        input_dim = embed_dim + pos_embed_dim + rule_embed_dim
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.projection = nn.Linear(input_dim, fusion_dim)

    def forward(self, word_embeds, pos_embeds, rule_embeds):
        """
        Args:
            word_embeds: [batch_size, seq_length, embed_dim]
            pos_embeds: [batch_size, seq_length, pos_embed_dim]
            rule_embeds: [batch_size, seq_length, rule_embed_dim]
        Returns:
            fused_embeds: [batch_size, seq_length, fusion_dim]
        """
        combined = torch.cat((word_embeds, pos_embeds, rule_embeds), dim=-1)  # [batch_size, seq_length, input_dim]
        gate_values = self.sigmoid(self.gate(combined))  # [batch_size, seq_length, input_dim]
        gated_combined = gate_values * combined  # Element-wise multiplication
        fused_embeds = self.projection(gated_combined)  # [batch_size, seq_length, fusion_dim]
        return fused_embeds


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using learnable embeddings.
    """
    def __init__(self, max_len, model_dim):
        super(PositionalEncoding, self).__init__()
        self.positional_embeddings = nn.Embedding(max_len, model_dim)

    def forward(self, x):
        """
        Adds positional encoding to input tensor x.
        Args:
            x: Tensor of shape [batch_size, seq_length, model_dim]
        Returns:
            Tensor with positional encoding added: [batch_size, seq_length, model_dim]
        """
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        pos_embeds = self.positional_embeddings(positions)
        return x + pos_embeds


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_length, model_dim]
            mask: [batch_size, seq_length] (optional)
        Returns:
            out: [batch_size, seq_length, model_dim]
        """
        batch_size, seq_length, _ = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_length, seq_length]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)  # [batch_size, num_heads, seq_length, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # [batch_size, seq_length, model_dim]
        out = self.fc_out(out)
        return out


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Network.
    """
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_length, model_dim]
        Returns:
            out: [batch_size, seq_length, model_dim]
        """
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = self.layer_norm(x + residual)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    """
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_length, model_dim]
            mask: [batch_size, seq_length] (optional)
        Returns:
            out: [batch_size, seq_length, model_dim]
        """
        # Self-Attention
        residual = x
        x = self.self_attention(x, mask)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        # Feed-Forward
        out = self.feed_forward(x)
        return out


class WordEncoder(nn.Module):
    """
    Encodes words within a sentence using Transformer Encoder Layers.
    """
    def __init__(self, embed_dim, pos_embed_dim, rule_embed_dim, fusion_dim,
                 model_dim, num_heads, ff_dim, num_layers, dropout=0.1, max_len=512):
        super(WordEncoder, self).__init__()
        self.fusion = FusionLayer(embed_dim, pos_embed_dim, rule_embed_dim, fusion_dim)
        self.position_encoding = PositionalEncoding(max_len, fusion_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(fusion_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_embeds, pos_embeds, rule_embeds, mask=None):
        """
        Args:
            word_embeds: [batch_size, seq_length, embed_dim]
            pos_embeds: [batch_size, seq_length, pos_embed_dim]
            rule_embeds: [batch_size, seq_length, rule_embed_dim]
            mask: [batch_size, seq_length] (optional)
        Returns:
            out: [batch_size, seq_length, fusion_dim]
        """
        x = self.fusion(word_embeds, pos_embeds, rule_embeds)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class WordAttention(nn.Module):
    """
    Applies attention over word embeddings to produce sentence representation.
    """
    def __init__(self, model_dim):
        super(WordAttention, self).__init__()
        self.attention = nn.Linear(model_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_length, model_dim]
            mask: [batch_size, seq_length] (optional)
        Returns:
            sentence_rep: [batch_size, model_dim]
        """
        scores = self.attention(x).squeeze(-1)  # [batch_size, seq_length]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, seq_length]
        sentence_rep = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, model_dim]
        return sentence_rep


class SentenceEncoder(nn.Module):
    """
    Encodes sentences within a document using Transformer Encoder Layers.
    """
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1, max_len=512):
        super(SentenceEncoder, self).__init__()
        self.position_encoding = PositionalEncoding(max_len, model_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence_embeds, mask=None):
        """
        Args:
            sentence_embeds: [batch_size, num_sentences, model_dim]
            mask: [batch_size, num_sentences] (optional)
        Returns:
            out: [batch_size, num_sentences, model_dim]
        """
        x = self.position_encoding(sentence_embeds)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SentenceAttention(nn.Module):
    """
    Applies attention over sentence embeddings to produce document representation.
    """
    def __init__(self, model_dim):
        super(SentenceAttention, self).__init__()
        self.attention = nn.Linear(model_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, num_sentences, model_dim]
            mask: [batch_size, num_sentences] (optional)
        Returns:
            document_rep: [batch_size, model_dim]
        """
        scores = self.attention(x).squeeze(-1)  # [batch_size, num_sentences]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_sentences]
        document_rep = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, model_dim]
        return document_rep


class HANTransformer(nn.Module):
    """
    Hierarchical Attention Network over Transformer architecture for document classification.
    """
    def __init__(self, vocab_size, pos_vocab_size, rule_vocab_size,
                 embed_dim, pos_embed_dim, rule_embed_dim, fusion_dim,
                 word_encoder_params, sentence_encoder_params,
                 num_classes, max_word_len=512, max_sent_len=64, max_rules_per_word=3):
        super(HANTransformer, self).__init__()

        # Embedding layers
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_dim, padding_idx=0)
        self.rule_embedding = nn.Embedding(rule_vocab_size, rule_embed_dim, padding_idx=0)

        # Word-level encoder and attention
        self.word_encoder = WordEncoder(
            embed_dim, pos_embed_dim, rule_embed_dim, fusion_dim, **word_encoder_params, max_len=max_word_len
        )
        self.word_attention = WordAttention(fusion_dim)

        # Sentence-level encoder and attention
        self.sentence_encoder = SentenceEncoder(
            **sentence_encoder_params, max_len=max_sent_len  # model_dim is included in sentence_encoder_params
        )
        self.sentence_attention = SentenceAttention(fusion_dim)

        # Classification head
        self.fc = nn.Linear(fusion_dim, num_classes)
        self.dropout = nn.Dropout(word_encoder_params.get('dropout', 0.1))

        # Other parameters
        self.max_rules_per_word = max_rules_per_word

    def forward(self, input_ids, attention_mask, pos_tags, rules, sentence_masks):
        """
        Args:
            input_ids: [batch_size, num_sentences, seq_length]
            attention_mask: [batch_size, num_sentences, seq_length]
            pos_tags: [batch_size, num_sentences, seq_length]
            rules: [batch_size, num_sentences, seq_length, max_rules_per_word]
            sentence_masks: [batch_size, num_sentences]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size, num_sentences, seq_length = input_ids.size()

        # Reshape for word-level processing
        input_ids = input_ids.view(-1, seq_length)  # [batch_size * num_sentences, seq_length]
        attention_mask = attention_mask.view(-1, seq_length)
        pos_tags = pos_tags.view(-1, seq_length)
        rules = rules.view(-1, seq_length, self.max_rules_per_word)

        # Word embeddings
        word_embeds = self.word_embedding(input_ids)  # [batch_size * num_sentences, seq_length, embed_dim]
        pos_embeds = self.pos_embedding(pos_tags)  # [batch_size * num_sentences, seq_length, pos_embed_dim]
        # Sum rule embeddings for each word
        rules_flat = rules.view(-1, seq_length * self.max_rules_per_word)
        rule_embeds = self.rule_embedding(rules_flat)  # [batch_size * num_sentences, seq_length * max_rules_per_word, rule_embed_dim]
        rule_embeds = rule_embeds.view(-1, seq_length, self.max_rules_per_word, rule_embeds.size(-1))
        rule_embeds = rule_embeds.sum(dim=2)  # [batch_size * num_sentences, seq_length, rule_embed_dim]

        # Word-level encoding
        word_encoded = self.word_encoder(word_embeds, pos_embeds, rule_embeds, attention_mask)
        # Word-level attention
        sentence_reps = self.word_attention(word_encoded, attention_mask)  # [batch_size * num_sentences, fusion_dim]

        # Reshape for sentence-level processing
        sentence_reps = sentence_reps.view(batch_size, num_sentences, -1)  # [batch_size, num_sentences, fusion_dim]

        # Sentence-level encoding
        sentence_encoded = self.sentence_encoder(sentence_reps, sentence_masks)
        # Sentence-level attention
        document_rep = self.sentence_attention(sentence_encoded, sentence_masks)  # [batch_size, fusion_dim]

        # Classification
        logits = self.fc(self.dropout(document_rep))  # [batch_size, num_classes]

        return logits


def get_model(vocab_size, pos_vocab_size, rule_vocab_size, num_classes,
              embed_dim=100, pos_embed_dim=25, rule_embed_dim=25, fusion_dim=128,
              word_encoder_params=None, sentence_encoder_params=None,
              max_word_len=512, max_sent_len=64, max_rules_per_word=3):
    """
    Utility function to initialize the HANTransformer model.
    """
    if word_encoder_params is None:
        word_encoder_params = {
            'model_dim': fusion_dim,
            'num_heads': 4,
            'ff_dim': fusion_dim * 4,
            'num_layers': 2,
            'dropout': 0.1,
        }
    if sentence_encoder_params is None:
        sentence_encoder_params = {
            'model_dim': fusion_dim,  # Ensure model_dim is included here
            'num_heads': 4,
            'ff_dim': fusion_dim * 4,
            'num_layers': 2,
            'dropout': 0.1,
        }

    model = HANTransformer(
        vocab_size=vocab_size,
        pos_vocab_size=pos_vocab_size,
        rule_vocab_size=rule_vocab_size,
        embed_dim=embed_dim,
        pos_embed_dim=pos_embed_dim,
        rule_embed_dim=rule_embed_dim,
        fusion_dim=fusion_dim,
        word_encoder_params=word_encoder_params,
        sentence_encoder_params=sentence_encoder_params,
        num_classes=num_classes,
        max_word_len=max_word_len,
        max_sent_len=max_sent_len,
        max_rules_per_word=max_rules_per_word
    )
    return model
