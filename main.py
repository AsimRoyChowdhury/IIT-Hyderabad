import matplotlib.pyplot as plt
from cifar.task import load_data

num_clients = 3

train_counts = []
test_counts = []

for cid in range(num_clients):
    _, _, train_size, test_size = load_data(cid, num_clients)
    train_counts.append(train_size)
    test_counts.append(test_size)

# Plotting
clients = [f"Client {i}" for i in range(num_clients)]
num = 1
for train_size, test_size in zip(train_counts, test_counts):
    print(f"Train size of Client {num}: {train_size}, Test size of Client {num}: {test_size}")
    num += 1

x = range(num_clients)
plt.bar(x, train_counts, width=0.4, label='Train Samples', align='center')
plt.bar(x, test_counts, width=0.4, label='Test Samples', align='edge')
plt.xticks(x, clients)
plt.ylabel("Number of Samples")
plt.title("Train/Test Samples per Client")
plt.legend()
plt.tight_layout()
plt.show()


from flwr.cli.run import run  # This is the actual callable

if __name__ == "__main__":
    run()