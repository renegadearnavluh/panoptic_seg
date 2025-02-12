import sys 
sys.path.insert(0, '../../my_ps')
from my_ps.loss.loss import MyLoss

if __name__=="__main__":
    loss = MyLoss()
    loss.say()