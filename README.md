# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 



### STEP 3: 



### STEP 4: 



### STEP 5: 



### STEP 6: 





## PROGRAM

### Name: BALA SARAVANAN K
### Register Number: 212224230031

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4) 

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')'

# Initialize model
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100) 

```

### Dataset Information
<img width="1313" height="341" alt="image" src="https://github.com/user-attachments/assets/39f1ac32-b271-48bc-a5ed-e1b13d8b573c" />


### OUTPUT
<img width="1029" height="395" alt="image" src="https://github.com/user-attachments/assets/062a69fe-9913-4964-a8d0-7133ba3a0690" />


## Confusion Matrix
<img width="676" height="572" alt="image" src="https://github.com/user-attachments/assets/846310bd-3a11-47de-a952-d6f3650cb719" />


## Classification Report
<img width="625" height="442" alt="image" src="https://github.com/user-attachments/assets/112c753f-6168-4937-9d5b-a5dd43806cf2" />


### New Sample Data Prediction
<img width="1029" height="395" alt="image" src="https://github.com/user-attachments/assets/062a69fe-9913-4964-a8d0-7133ba3a0690" />

## RESULT
Thus, a neural network Classification model was successfully developed and trained using PyTorch.
