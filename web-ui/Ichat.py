import torch
from AI.demo.AIDemo import Nltk_utils, nlpp, NerualNet
import random

hidden_size = 8
output_size = len(nlpp.tags)
input_size = len(nlpp.X_train[0])
model = NerualNet(input_size, hidden_size, output_size)

checkpoint = torch.load("data.pth")
model.load_state_dict(checkpoint["model_state"])


def answer(sentence, bot_name = "Nora"):
    n = Nltk_utils()
    sentence = n.tokenize(sentence)
    X = n.bag_of_words(sentence, nlpp.all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = nlpp.tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in nlpp.intents['intents']:
            if tag == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
                return f'{random.choice(intent["responses"])}'
    else:
        print(f'{bot_name}: I dont understand...')
        return None



def chat():

    print("chat started.")
    while True:
        sentence = input("say something: ")
        if sentence == "quit":
            break

        answer(sentence)



if __name__ == "__main__":
    chat()
