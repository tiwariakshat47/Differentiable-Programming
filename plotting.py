import matplotlib.pyplot as plt

def plot_losses_wrt_epoch(losses):
    # Graph losses and offsets as separate lines wrt time.
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(losses, color = 'g')
    plt.title("Loss Curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss", color = 'g')
    plt.show()

def plot_losses_offsets_wrt_epoch(losses, offsets):
    # Graph losses and offsets as separate lines wrt time.
    fig, ax = plt.subplots(figsize = (10, 5))
    ax2 = plt.twinx()
    ax.plot(losses, color = 'g')
    ax2.plot(offsets, color = 'r')
    plt.title("Loss and Offset Curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss", color = 'g')
    ax2.set_ylabel("Offset", color = 'r')
    plt.show()

def plot_losses_wrt_param(losses, params, title="Losses"):
    # Graph losses wrt offsets.
    plt.plot(losses, params)
    plt.title(title)
    plt.xlabel("Param")
    plt.ylabel("Loss")
    plt.show()

def plot_multiple_losses_wrt_param(params, losses_dict, title="Losses"):
    for name, losses in losses_dict.items():
        plt.plot(params, losses, label=name)
    plt.title(title)
    plt.xlabel("Param")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_losses_sigmas(offsets, sigma_losses, title='Loss'):
    # Graph multiple loss for multiple sigma values wrt to offset.
    [plt.plot(offsets, losses, label=f"sigma={str(sigma.item())[:4]}")
     for sigma, losses in sigma_losses.items()]

    plt.title(title)
    plt.xlabel("Offset")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()