import random
import pickle

with open('text.pkl', 'rb') as f:
    texts = pickle.load(f)

random_seed = input('Enter a random seed: ')
random_seed = int(random_seed)
random_seed = random_seed % int(1e9+7)
random.seed(random_seed)

n = len(texts)
text = texts[random.randint(0, n-1)]
ea = [i for i in range(13)]
random.shuffle(ea)
ea = ea+[i for i in range(13,26)]
text = [ chr(ea[ord(text[i])-97]+97) for i in range(len(text))]
text = ''.join(text)
print(text)