import matplotlib.pyplot as plt


def draw_figure(images, size=(2, 2), n=[0]):
    # 2 x 3 figure
    fig, ax = plt.subplots(size[0], size[1], dpi = 300)
    index = 0
    for i in range(size[0]):
        for j in range(size[1]):
            try:
                ax[i, j].imshow(images[i*size[1]+j])
                ax[i, j].text(0, 0, f'frame {n[index]}', color='black', fontsize=8)
                index += 1
                ax[i, j].axis('off')
            except IndexError:
                ax[i, j].axis('off')
    # save figure
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig('figure.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    n = [0, 140, 160, 184]
    images = [plt.imread(f'./outputs/frame_{i}.jpg') for i in n]
    draw_figure(images, (2, 2), n)
            
