import torch

class train_model():
    def __init__(self,model,train_loader,val_loader,optimizer,criterion,n_test,n_epochs,path):
        self.PATH = path # "entire_model.pt"
        self.N_test = n_test
        self.accuracy_list = []
        self.loss_list = []
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.n_epochs = n_epochs

    def train(self):
        for epoch in range(self.n_epochs):
            print(f"Epoch - {epoch} Started")
            for x, y in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                z = self.model(x)
                loss = self.criterion(z,y)
                loss.backward()
                self.optimizer.step()
                self.loss_list.append(loss)

            correct = 0
            for x_test, y_test in self.val_loader:
                self.model.eval()
                z = self.model(x_test)
                _, yhat = torch.max(z.data,1)
                correct += (yhat == y_test).sum().item()
            accuracy = correct/self.N_test
            self.accuracy_list.append(accuracy)
            print(f"Epoch - {epoch} Ended")

        torch.save(self.model,self.PATH)
        return self.accuracy_list, self.loss_list