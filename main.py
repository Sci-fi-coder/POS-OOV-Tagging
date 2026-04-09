from src.data_loader import load_conllu
from src.vocab import Vocab
from src.dataset import POSDataset
from src.train import train_model
from src.evaluate import evaluate
from src.predict import predict_sentence
from model.pos_model import POSTagger


# ==============================
# 1. Load Dataset
# ==============================

train_sents, train_tags = load_conllu("data/en_ewt-ud-train.conllu")
test_sents, test_tags = load_conllu("data/en_ewt-ud-test.conllu")


# ==============================
# 2. Build Vocabulary
# ==============================

vocab = Vocab()
vocab.build(train_sents, train_tags)


# ==============================
# 3. Prepare Dataset
# ==============================

dataset = POSDataset(train_sents, train_tags, vocab)


# ==============================
# 4. Initialize Model
# ==============================

model = POSTagger(vocab)


# ==============================
# 5. Train Model (increased epochs for better accuracy)
# ==============================

train_model(model, dataset, vocab, epochs=5)


# ==============================
# 6. Evaluate Model
# ==============================

evaluate(model, test_sents, test_tags, vocab)


# ==============================
# 7. Interactive Demo
# ==============================

print("\n----- Interactive POS Tagging -----")
print("Type a sentence to test OOV handling.")
print("Type 'exit' to stop.\n")

while True:

    sentence = input("Enter sentence: ")

    if sentence.lower() == "exit":
        break

    predict_sentence(model, sentence, vocab)