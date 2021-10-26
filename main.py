from __future__ import division
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from utils import *
from model import NeuralNetwork

from tensorboardX import SummaryWriter
import shutil
from random import shuffle

log_dir = 'log_train'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(logdir=log_dir, flush_secs=10)

model_path = 'backup/model.pth'

# Get cpu or gpu device for training.
if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = "cpu"
print("[Using {} device]".format(device))


# training parameters configuration
training_epochs = 200
learning_rate_current = 0.001

test_step = 5

is_resume_model = False
is_resume_data = True

# define model, loss function and optimizer
model = NeuralNetwork().to(device)

if is_resume_model:
    model.load_state_dict(torch.load(model_path))
    print('load model successfully!')

# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

# Channel conditions dataset
channel_response_set_train = []
channel_response_set_test = []

if is_resume_data:
    channel_response_set_train = np.load('channel_response_set_train.npy')
    channel_response_set_test = np.load('channel_response_set_test.npy')
    print('[channel data loaded successfully!]')
else:
    # The H information set
    H_folder_train = './H_dataset/'
    H_folder_test = './H_dataset/'
    train_idx_low = 1
    train_idx_high = 301
    test_idx_low = 301
    test_idx_high = 401

    # Saving Channel conditions to a large matrix
    for train_idx in range(train_idx_low, train_idx_high):
        print("Processing the ", train_idx, "th document")
        H_file = H_folder_train + str(train_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(
                    numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_train.append(h_response)

    for test_idx in range(test_idx_low, test_idx_high):
        print("Processing the ", test_idx, "th document")
        H_file = H_folder_test + str(test_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(
                    numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_test.append(h_response)

    np.save('channel_response_set_train.npy', channel_response_set_train)
    np.save('channel_response_set_test.npy', channel_response_set_test)
    print('channel data saved successfully!')
    print('length of training channel response', len(channel_response_set_train),
            'length of testing channel response', len(channel_response_set_test))


def train(epoch):
    loss = 0
    total_loss = 0
    
    train_channel_set_size = len(channel_response_set_train)
    train_channel_set_idx = np.arange(train_channel_set_size)
    shuffle(train_channel_set_idx)

    batch_size = 500  # 一个batch中的样本数
    total_batch = int(train_channel_set_size/batch_size)  # 一个epoch中的batch数

    model.train()

    for i in range(total_batch):
        input_samples = []
        input_labels = []

        for j in range(batch_size):
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
            channel_response = channel_response_set_train[train_channel_set_idx[i * batch_size + j]]
            signal_output, para = ofdm_simulate(bits, channel_response, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)

        batch_x = torch.from_numpy(np.array(input_samples))
        batch_y = torch.from_numpy(np.array(input_labels))

        data, target = batch_x.to(device).float(), batch_y.to(device)

        # Compute prediction error
        pred = model(data)
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        loss = loss_fn(pred, target)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"{epoch} \t {i+1}/{total_batch} \t loss: {loss.item():>7f}")
    avg_loss = total_loss/total_batch
    writer.add_scalar(tag='avg_loss', scalar_value=avg_loss, global_step=epoch)
    print(f"epoch: {epoch}       avg_loss: {avg_loss:>7f}")
    return avg_loss


def test(epoch):
    test_channel_set_size = len(channel_response_set_test)
    batch_size = 300  # 一个batch中的样本数
    total_batch = int(test_channel_set_size/batch_size)  # 一个epoch中的batch数

    model.eval()

    total_ser = 0
    best_ser = 10 # 随机设置的一个较大值

    with torch.no_grad():
        for i in range(total_batch):
            input_samples = []
            input_labels = []

            for j in range(batch_size):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                channel_response = channel_response_set_test[i * batch_size + j]
                signal_output, para = ofdm_simulate(bits, channel_response, SNRdb)
                input_labels.append(bits[0:16])
                input_samples.append(signal_output)

            batch_x = torch.from_numpy(np.array(input_samples))
            batch_y = torch.from_numpy(np.array(input_labels))

            data, target = batch_x.to(device).float(), batch_y.to(device)

            # Compute prediction error
            pred = model(data)

            pred = torch.sign(pred - 0.5)
            target = torch.sign(target - 0.5)

            total_ser += torch.mean((pred != target).float()).item()
        avg_ser = total_ser / total_batch
        print(f"\n\ntest result  -->  avg_ser: {avg_ser:>7f} \n\n")
        writer.add_scalar(tag='avg_ser', scalar_value=avg_ser, global_step=epoch)

        if avg_ser < best_ser:
            torch.save(model.state_dict(), model_path)
            best_ser = avg_ser
            print(f'save best model in >>> {model_path}')
        return avg_ser


loss_final = 0
ber_final = 0
for t in range(training_epochs):
    t += 1

    loss_final = train(t)
    if t % test_step == 0:
        ber_final = test(t)

writer.close()

print("final loss:", loss_final)
print("final ber:", ber_final)