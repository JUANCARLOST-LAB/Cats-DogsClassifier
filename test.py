from model import model
from main import validation_iterator

acc = model.evaluate(validation_iterator)

print(acc)