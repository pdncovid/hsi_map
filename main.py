from fn.data_functions import samples_get, samples_zeros
from fn.map_functions import load_maps

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
file = 'imageKandy'


def main():
    [stacked_arr, _] = load_maps(folder, file, view=False, out_array=True)
    rows = 50
    data_sampled = samples_get(stacked_arr[0], rows, view=True, print_=True)
    samples_zeros(data_sampled, view=True, print_=True)


if __name__ == "__main__":
    main()
