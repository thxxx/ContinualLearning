import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import tqdm
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from datasets import load_dataset
from utils import CNN

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="if already pre-trained model is saved.",
    )
    parser.add_argument(
        "--btach_size",
        type=int,
        default=32,
        required=True,
        help="batch size of training",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    batch_size=args.batch_size

    class CifarDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            super(CifarDataset, self).__init__()
            self.data_list=[]
            for data in tqdm.tqdm(dataset):
                self.data_list.append({
                    "img":transform(data['img']),
                    "label":data['fine_label']
                    })
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            return self.data_list[idx]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    cifar_dataset = load_dataset("cifar100")

    cifar_train=CifarDataset(cifar_dataset['train'])
    cifar_test=CifarDataset(cifar_dataset['test'])
    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True, num_workers=2)
    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True, num_workers=2)

    training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # train_10_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_10_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    # 전체 데이터셋을 태스크별로 분리

    tasks=[[], [], [], [], []]
    batch_size=16

    for data in tqdm.tqdm(training_dataset):
        tasks[data[1]//2].append(data)
    
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, task_num):
            super(TaskDataset, self).__init__()
            self.data_list=tasks[task_num]
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            return self.data_list[idx]

    task1_dataloader = torch.utils.data.DataLoader(TaskDataset(0), batch_size=batch_size, shuffle=True, num_workers=2)
    task2_dataloader = torch.utils.data.DataLoader(TaskDataset(1), batch_size=batch_size, shuffle=True, num_workers=2)
    task3_dataloader = torch.utils.data.DataLoader(TaskDataset(2), batch_size=batch_size, shuffle=True, num_workers=2)
    task4_dataloader = torch.utils.data.DataLoader(TaskDataset(3), batch_size=batch_size, shuffle=True, num_workers=2)
    task5_dataloader = torch.utils.data.DataLoader(TaskDataset(4), batch_size=batch_size, shuffle=True, num_workers=2)

    device="cpu"
    if torch.cuda.is_available():
        device="cuda"

    model = CNN()
    model.to(device)

    ### training
    # Set up training loop
    class ContinualLearning:
        def __init__(self, learning_rate=0.01):
            self.train_logs={
                "losses":[],
                "pretrain_losses":[],
                "validation_accuracies":[],
            }
            self.running_loss=0
            self.criterion=nn.CrossEntropyLoss()
            self.learning_rate = 0.01

        def pretrain(self, epochs=5, learning_rate=0.01, dataloader=cifar_train_loader):
            self.learning_rate = learning_rate
            # model.conv, model.fc_layer 둘다 학습
            model.conv.requires_grad_(True)
            model.fc_layer.requires_grad_(True)
            model.train()

            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            timed=time.time()
            for epoch in range(epochs):
                self.running_loss = 0.0
                print(f"epoch : {epoch} is running...")
                for idx, data in enumerate(tqdm.tqdm(dataloader)):
                    optimizer.zero_grad()
                    img, label = data['img'], data['label']
                    img=img.to(device)
                    label=label.to(device)

                    outputs = model(img)

                    loss = self.criterion(outputs, label)
                    loss.backward()
                    optimizer.step()

                    self.running_loss += loss.cpu().detach().item()
                print(f'\n {time.time() - timed}초 걸림 [{epoch + 1}, {idx+1}] average loss: {self.running_loss / 500}')
                timed=time.time()
                self.train_logs['pretrain_losses'].append(self.running_loss / 500)
                self.running_loss = 0.0
                
                # torch.save(model.state_dict(), GOOGLE_DRIVE_PATH + "pretrained.pt")
                model.eval()
                self.pretrain_validate()

        def pretrain_validate(self):
            class_correct = list(0. for i in range(100))
            class_total = list(0. for i in range(100))
            with torch.no_grad():
                for idx, data in enumerate(tqdm.tqdm(cifar_test_loader)):
                    images, labels = data['img'], data['label']
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # print("outputs: ", outputs.shape, labels)

                    _, predicted = torch.max(outputs, 1)
                    # if idx%10==0:
                    #   print("predicted : :", predicted)

                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                    if idx%500==499:
                        print("validating...")

            for i in range(100):
                print('Accuracy of %5s : %2d %%' % (
                    i, 100 * class_correct[i] / class_total[i]))
            plt.plot(self.train_logs['pretrain_losses'])
            plt.show()

        def continue_train(self, epochs=5, learning_rate=0.001):
            self.learning_rate = learning_rate
            # model.conv는 fix시키고, fc_layer만 학습
            model.requires_grad_(False)
            model.eval()

            timed=time.time()
            for idx, train_dataloader in enumerate([task1_dataloader, task2_dataloader, task3_dataloader, task4_dataloader, task5_dataloader]):
                fc_layer = nn.Sequential(
                    nn.Linear(32*16*16, 2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 2),   
                    ).to(device)
                fc_layer = fc_layer.requires_grad_(True)
                optimizer = torch.optim.SGD(fc_layer.parameters(), lr=self.learning_rate)
                fc_layer.train()

                print(f"{idx}th task is being trained...")
                for epoch in tqdm.tqdm(range(epochs)):
                    self.running_loss = 0.0

                    for i, data in enumerate(train_dataloader):
                        optimizer.zero_grad()
                    
                        inputs, labels = data
                        inputs=inputs.to(device)
                        labels=labels.to(device)

                        features = model(inputs, conv=True)
                        outputs = fc_layer(features)

                        outputs = torch.cat((torch.zeros([16,idx*2]).to(device)-10, outputs), dim=1)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        # Print statistics
                        self.running_loss += loss.cpu().detach().item()
                        print(f'\n {time.time() - timed}초 걸림 [{epoch + 1}, {i+1}] loss: {self.running_loss / 300}')
                        timed=time.time()
                        self.train_logs['losses'].append(self.running_loss / 300)
                        self.running_loss = 0.0
                fc_layer.eval()
                fc_layer.requires_grad_(False)
                model.fc_layers.append(fc_layer)
                self.visualize_losses()
            # self.validate(model, valid_10_dataloader)
            # plt.plot(self.train_logs['validation_accuracies'])
            # plt.ylim(0, 100)
            # plt.grid(True)
            # plt.xticks([0, 1, 2, 3, 4], ['1', "2", "3", "4", "5"])
            # plt.xlabel("Task")
            # plt.show()
        
        def validate(self, model, test_data_loader):
            model.eval()
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            class_correct2 = list(0. for i in range(10))
            with torch.no_grad():
                for idx, data in enumerate(tqdm.tqdm(test_data_loader)):
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)

                    vals_list, idx_list, diffs = model.continues(images)
                    ll=[]
                    for i, v in enumerate(torch.max(torch.stack(vals_list), dim=0).indices):
                        last_label = v*2 + idx_list[v][i]
                        ll.append(last_label)

                    ll2=[]
                    for i, v in enumerate(torch.max(torch.stack(diffs), dim=0).indices):
                        last_label = v*2 + idx_list[v][i]
                        ll2.append(last_label)

                    # save accuracies
                    for i in range(len(labels.cpu().detach().tolist())):
                        label = labels.cpu().detach().tolist()[i]
                        class_total[label] += 1
                        if label == [r.cpu().detach().item() for r in ll][i]:
                            class_correct[label] += 1
                        if label == [r.cpu().detach().item() for r in ll2][i]:
                            class_correct2[label] += 1
                        
                    if idx%500==499:
                        print("validating...")

                for i in range(10):
                    print('Accuracy of %5s : %2d %%' % (
                        i, 100 * class_correct[i] / class_total[i]))
                print("\n\n")
                for i in range(10):
                    print('Accuracy of based on different. %5s : %2d %%' % (
                        i, 100 * class_correct2[i] / class_total[i]))
                self.train_logs['validation_accuracies'].append(sum(class_correct2)/10)
        
        def visualize_losses(self):
            plt.plot(self.train_logs['losses'])
            plt.show()
    
    cont = ContinualLearning()

    ## 사전학습
    cont.pretrain(learning_rate=0.01, epochs=20, dataloader=cifar_train_loader)

    ## 연속학습
    cont.continue_train(epochs=10, learning_rate=0.005)

    ## validate
    cont.validate(model, valid_10_dataloader) # cont.validate(model, test_data_loader)

if __name__ == "__main__":
    main()