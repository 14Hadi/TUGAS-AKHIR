from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import re
import random
import math
from collections import Counter
from gensim.models import Word2Vec

# Definisikan model dan fungsi-fungsi yang diperlukan
class RNNWithAttention(nn.Module):
    def __init__(self, embedding_matrix_src, embedding_matrix_tgt, hidden_dim=512, n_layers=2):
        super(RNNWithAttention, self).__init__()
        embedding_dim = embedding_matrix_src.size(1)
        self.embedding_src = nn.Embedding.from_pretrained(embedding_matrix_src, freeze=False)
        self.embedding_tgt = nn.Embedding.from_pretrained(embedding_matrix_tgt, freeze=False)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, embedding_matrix_tgt.size(0))
        
        # Linear layers untuk mengurangi dimensi dari 1024 menjadi 512
        self.reduce_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # Embedding
        embedded_src = self.embedding_src(src)
        embedded_tgt = self.embedding_tgt(tgt)

        # Encode urutan sumber
        encoder_output, (hidden, cell) = self.encoder(embedded_src)

        # Sesuaikan hidden dan cell states: Kombinasikan states dari kedua arah
        hidden = hidden.view(self.encoder.num_layers, 2, hidden.size(1), hidden.size(2))
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        
        cell = cell.view(self.encoder.num_layers, 2, cell.size(1), cell.size(2))
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

        # Kurangi dimensi kembali ke hidden_dim
        hidden = self.reduce_hidden(hidden)
        cell = self.reduce_cell(cell)

        # Inisialisasi output tensor
        batch_size = tgt.size(0)
        max_len = tgt.size(1)
        outputs = torch.zeros(batch_size, max_len, self.fc_out.out_features).to(tgt.device)

        # Input pertama ke decoder adalah token <sos>
        input = tgt[:, 0].unsqueeze(1)

        for t in range(1, max_len):
            # Decode urutan target token demi token
            output, (hidden, cell) = self.decoder(self.embedding_tgt(input), (hidden, cell))

            # Prediksi token berikutnya
            output = self.fc_out(output.squeeze(1))
            outputs[:, t] = output

            # Tentukan apakah menggunakan teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Ambil token yang diprediksi tertinggi atau token berikutnya yang sebenarnya
            top1 = output.argmax(1).unsqueeze(1)
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs

def case_folding(text):
    return text.lower()

def cleansing(text):
    text = text.replace('-', ' ')
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Izinkan hanya huruf dan spasi
    text = re.sub(r'\s+', ' ', text)  # Hapus spasi ganda
    return text.strip()

def tokenize(text):
    if isinstance(text, str):
        return text.split()
    else:
        raise ValueError(f"Input harus berupa string, tetapi didapat {type(text)}")

def encode_text(tokenized_text, word_to_index):
    encoded_text = [word_to_index['<sos>']] + [word_to_index.get(word, word_to_index['<unk>']) for word in tokenized_text]
    encoded_text.append(word_to_index['<eos>'])  # Tambahkan <eos>
    return encoded_text

def decode_text(encoded_text, index_to_word, pad_idx, unk_idx, eos_idx, sos_idx):
    tokens = [index_to_word.get(token, '<unk>') for token in encoded_text if token not in (pad_idx, unk_idx, eos_idx, sos_idx)]
    decoded_sentence = ' '.join(tokens).strip()
    return decoded_sentence

def clean_sentence(sentence):
    return ' '.join(sentence.replace('<eos>', '').replace('<unk>', '').replace('<sos>', '').replace('<pad>', '').split())

def translate_sentence(model, sentence, source_word_to_index, target_index_to_word, max_length=50, device='cpu'):
    # Praproses kalimat
    sentence = case_folding(sentence)
    sentence = cleansing(sentence)
    tokenized_sentence = tokenize(sentence)
    encoded_sentence = encode_text(tokenized_sentence, source_word_to_index)
    
    # Ubah menjadi tensor dan tambahkan dimensi batch
    src_tensor = torch.LongTensor(encoded_sentence).unsqueeze(0).to(device)

    # Inisialisasi urutan target dengan <sos>
    tgt_tensor = torch.LongTensor([target_word_to_index['<sos>']]).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            output = model(src_tensor, tgt_tensor)
            pred_token = output.argmax(2)[:, -1].item()

            # Berhenti jika model memprediksi <eos>
            if pred_token == target_word_to_index['<eos>']:
                break
            
            # Tambahkan token yang diprediksi ke urutan target
            tgt_tensor = torch.cat([tgt_tensor, torch.LongTensor([[pred_token]]).to(device)], dim=1)
        
        # Dekode urutan output
        pred_text = decode_text(tgt_tensor.squeeze(0).cpu().numpy(), target_index_to_word, 
                                pad_idx=source_word_to_index['<pad>'], 
                                unk_idx=source_word_to_index['<unk>'], 
                                eos_idx=source_word_to_index['<eos>'], 
                                sos_idx=source_word_to_index['<sos>'])
        
        # Bersihkan kalimat hasil dekode
        cleaned_pred_text = clean_sentence(pred_text)
    
    return cleaned_pred_text

# Load model dan data lainnya
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('nmt_muna_model.pth', map_location=device)
source_word_to_index = checkpoint['source_word_to_index']
target_word_to_index = checkpoint['target_word_to_index']
source_index_to_word = checkpoint['source_index_to_word']
target_index_to_word = checkpoint['target_index_to_word']
source_embedding_matrix = checkpoint['source_embedding_matrix']
target_embedding_matrix = checkpoint['target_embedding_matrix']

model = RNNWithAttention(source_embedding_matrix, target_embedding_matrix).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Membuat API dengan Flask
app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    sentence = data.get('sentence', '')

    if not sentence:
        return jsonify({'error': 'Kalimat tidak boleh kosong'}), 400

    translated_sentence = translate_sentence(model, sentence, source_word_to_index, target_index_to_word, device=device)
    return jsonify({'translated_sentence': translated_sentence})

if __name__ == '__main__':
    app.run(debug=True)
