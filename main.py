import torch
import torch.nn as nn
import torch.optim as optim
from torch._C._nn import gelu
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertLayer, BertAttention, BertSdpaSelfAttention, BertSelfOutput, BertIntermediate, BertOutput, BertConfig, BertPooler
from sklearn.model_selection import train_test_split
import json
import os
import torch.serialization

# --- 1. データ準備 ---

class GreetingDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["target"]

        input_encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "target_ids": target_encoding["input_ids"].squeeze(),
            "target_mask": target_encoding["attention_mask"].squeeze()
        }

def load_greeting_data(file_path):
    """JSONファイルから挨拶データをロード"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_datasets(data, tokenizer, test_size=0.2, max_len=128):
    """データセットを分割"""
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    train_dataset = GreetingDataset(train_data, tokenizer, max_len)
    val_dataset = GreetingDataset(val_data, tokenizer, max_len)
    return train_dataset, val_dataset


# --- 2. モデル定義 ---
class GreetingModel(nn.Module):
    def __init__(self, bert_model_name, output_size, max_len):
        super(GreetingModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.decoder = nn.Linear(self.bert.config.hidden_size, output_size)
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state  # [batch_size, max_len, hidden_size]
        decoded_output = self.decoder(last_hidden_states) # [batch_size, max_len, output_size]
        return decoded_output

def load_or_create_model(model_path, bert_model_name, tokenizer, output_size, max_len):
    """モデルが存在すればロード、なければ新規作成"""
    if os.path.exists(model_path):
      model = GreetingModel(bert_model_name, output_size, max_len) # モデルの構造をまず作成
      with torch.serialization.safe_globals([GreetingModel, BertModel, BertEmbeddings, BertLayer, BertAttention, BertSdpaSelfAttention, nn.Embedding, nn.LayerNorm, nn.Dropout, BertEncoder, nn.ModuleList, nn.Linear, BertSelfOutput, BertIntermediate, GELUActivation, gelu, BertOutput, BertConfig, BertPooler, nn.Tanh]): # ModuleList を追加
        with open(model_path, 'rb') as f: # 重みだけを読み込む
          state_dict = torch.load(f, map_location=torch.device('cpu'))
          model.load_state_dict(state_dict) # 保存した重みをロード
      print("Loaded model from disk.")
    else:
      model = GreetingModel(bert_model_name, output_size, max_len)
      print("Created new model.")
    return model

# --- 3. 学習処理 ---

def train_model(model, train_dataset, val_dataset, tokenizer, learning_rate=2e-5, batch_size=32, epochs=1, model_path="greeting_model.pth"):
    """モデルの学習"""
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() # 応答の単語の確率分布を学習するため
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask) # [batch_size, max_len, output_size]
            
            # ロス計算のためにターゲットを整える
            target_ids_reshaped = target_ids.view(-1) # [batch_size * max_len]
            outputs_reshaped = outputs.view(-1, outputs.shape[-1]) # [batch_size * max_len, output_size]
            loss = criterion(outputs_reshaped, target_ids_reshaped)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch+1}, Training Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path) # モデルの重みだけを保存する
    print(f"Saved model to {model_path}")

# --- 4. 推論処理テスト ---

def eval_model(model, train_dataset, val_dataset, tokenizer, learning_rate=2e-5, batch_size=32, epochs=1, model_path="greeting_model.pth"):
    """モデルの学習"""
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss() # 応答の単語の確率分布を学習するため
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.eval() # 検証データでの評価
        eval_loss = 0
        with torch.no_grad():
          for batch in val_dataloader:
              input_ids = batch["input_ids"].to(device)
              attention_mask = batch["attention_mask"].to(device)
              target_ids = batch["target_ids"].to(device)

              outputs = model(input_ids, attention_mask)
              target_ids_reshaped = target_ids.view(-1) # [batch_size * max_len]
              outputs_reshaped = outputs.view(-1, outputs.shape[-1])  # [batch_size * max_len, output_size]
              loss = criterion(outputs_reshaped, target_ids_reshaped)
              eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(val_dataloader)
        print(f"Epoch: {epoch+1}, Validation Loss: {avg_eval_loss:.4f}")


# --- 5. 推論処理 ---
def generate_response(model, tokenizer, input_text, max_len, device):
    """応答の生成"""
    input_encoding = tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
    )

    model.eval() # 推論モードに変更
    with torch.no_grad():
      input_ids = input_encoding["input_ids"].to(device)
      attention_mask = input_encoding["attention_mask"].to(device)
      outputs = model(input_ids, attention_mask)
      
    predicted_token_ids = torch.argmax(outputs, dim=-1).cpu().numpy()
    response = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True) # バッチサイズが1なので最初の要素を取り出す

    return response

# --- 5. 実行スクリプト ---

def main():
    # 設定
    DATA_FILE = "test.json"  # JSONデータファイル名
    MODEL_PATH = "gren.pth" # モデル保存ファイル名
    BERT_MODEL_NAME = "bert-base-uncased" # 利用するBERTモデル名
    MAX_LEN = 128 # 最大トークン長
    OUTPUT_SIZE = 30522 # BERTのvocabサイズ
    
    # トークナイザーの準備
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    # モデルの準備（初回学習 or 追加学習）
    model = load_or_create_model(MODEL_PATH, BERT_MODEL_NAME, tokenizer, OUTPUT_SIZE, MAX_LEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # データの準備
    if not os.path.exists(DATA_FILE):
      # テスト用のダミーデータを作成
      dummy_data = [
        {"input": "こんにちは", "target": "こんにちは"},
        {"input": "おはよう", "target": "おはようございます"},
        {"input": "こんばんは", "target": "こんばんは"},
        {"input": "ありがとう", "target": "どういたしまして"},
        {"input": "さようなら", "target": "さようなら"},
        {"input": "exit", "target": "ご利用ありがとうございました。"}
      ]
      with open(DATA_FILE, 'w', encoding='utf-8') as f:
          json.dump(dummy_data, f, ensure_ascii=False, indent=2)
    
    data = load_greeting_data(DATA_FILE)
    train_dataset, val_dataset = create_datasets(data, tokenizer, max_len=MAX_LEN)

    # エポック数の設定
    set_epochs_check = input("エポック数を設定しますか？(y/N): ")
    if set_epochs_check.lower() == "y":
      set_epochs_text = input("エポック数を入力してください: ")
      set_epochs = int(set_epochs_text)
    else:
      set_epochs = 5


    # 学習
    if not os.path.exists(MODEL_PATH):
      print("モデルの初回学習します")
      train_model(model, train_dataset, val_dataset, tokenizer, epochs = set_epochs, model_path=MODEL_PATH)
    else:
      print("モデルはすでに学習済みです。必要であれば追加学習を行います")
      additional_train = input("追加学習しますか？(Y/n): ")
      if additional_train.lower() == "n":
        print("ではモデルを初期化しますか？")
        initialize = input("初期化しますか？(y/N): ")
        if initialize.lower() == "y":
          print("モデルを初期化します")
          exec('rm gren.pth')
          main()
        else:
          print("学習を終了します")
          exit()
      else:
        print("追加学習を開始します")
        train_model(model, train_dataset, val_dataset, tokenizer, epochs = set_epochs, model_path=MODEL_PATH)


    # 推論テスト
    print("推論テストを開始します")
    eval_model(model, train_dataset, val_dataset, tokenizer, epochs = 1, model_path=MODEL_PATH)


    # 推論実行確認
    ai_start = input("AIを起動しますか？(y/N): ")
    if ai_start.lower() != "y":
      print("AI学習を終了します")
      exit()

    # AIの名前設定
    ai_name = input("私の名前を入力してください: ")
    print(f"{ai_name}: こんにちは！ 要件を入力してください")

    # 推論
    while True:
      input_text = input("(終了するにはexit): ")
      response = generate_response(model, tokenizer, input_text, MAX_LEN, device)
      print(f"{ai_name}: {response}")
      if input_text.lower() == "exit":
          break

if __name__ == "__main__":
    main()