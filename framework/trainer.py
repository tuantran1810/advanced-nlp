import os
import torch
from loguru import logger as log

class Trainer:
    def __init__(
        self, 
        model_name, 
        inject_dataloader, 
        inject_model, 
        inject_optim, 
        inject_loss_fn, 
        inject_accuracy_calculator,
        log_per_samples = 1000,
    ):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model_name = model_name
        self.__train_loader, self.__test_loader = inject_dataloader()
        self.__model = inject_model().to(self.__device)
        self.__optim = inject_optim(self.__model)
        self.__loss = inject_loss_fn()
        self.__accuracy = inject_accuracy_calculator()
        self.__log_per_samples = log_per_samples

    def get_model(self):
        return self.__model

    def train(self, epochs, model_path=""):
        if model_path == "":
            model_path = "./model-{}".format(self.__model_name)
        log.info("model will be saved to: {}".format(model_path))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        for i in range(epochs):
            log.info(f"---------------------epoch {i}---------------------")
            self.__model.train()
            totalloss = 0
            cnt = 0
            for x, y in self.__train_loader:
                cnt += 1
                x = x.to(self.__device)
                y = y.to(self.__device)

                self.__optim.zero_grad()
                yhat = self.__model(x)
                loss = self.__loss(yhat, y)
                loss.backward()
                self.__optim.step()
                with torch.no_grad():
                    totalloss += loss
                    if cnt % self.__log_per_samples == 0:
                        accuracy = self.__accuracy(yhat, y)*100
                        log.info("log per {} samples, {} samples has passed, training loss: {:.5f}, training accuracy: {:.5f}%".format(self.__log_per_samples, cnt, totalloss/cnt, accuracy))
            with torch.no_grad():
                log.info("training loss: {:.5f}".format(totalloss/cnt))
                torch.save(self.__model, model_path + "/{}.pt".format(i))

            self.__model.eval()
            with torch.no_grad():
                totalloss = 0
                totalaccuracy = 0
                cnt = 0
                totalsamples = 0

                for x, y in self.__test_loader:
                    cnt += 1
                    totalsamples += len(x)
                    x = x.to(self.__device)
                    y = y.to(self.__device)
                    yhat = self.__model(x)
                    loss = self.__loss(yhat, y)
                    totalloss += loss
                    totalaccuracy += self.__accuracy(yhat, y)

                log.info("loss = {:.5f}, accuracy = {:.5f}%".format(totalloss/cnt, (totalaccuracy/cnt) * 100))
