import os
import string
import pandas as pd
import geopy.distance

# Paths of all the datasets
dataset_path = os.getcwd()+'\\data\\pedestrian'
labeled_data_path = os.getcwd()+'\\data\\Label'



# Define a function to convert numbers into corresponding letter labels
def num_to_letter(num):
    '''
    num         : number that we have to convert
    '''
    return string.ascii_uppercase[num]

# Define a function to check whether the path passes through the grid
def is_path_in_grid(south, west, north, east, path_points):
    '''
    south       : minimum latitude
    west        : minimum longitude
    north       : maximum latitude
    east        : maximum longitude
    path_points : coordinate points
    '''
    for lat, lng in path_points:
        if south <= lat <= north and west <= lng <= east:
            return True
    return False

# Create a function to get the grid label of the coordinate point
def get_grid_label(lat, lng, final_grids):
    '''
    lat         : latitude
    lng         : longitude
    final_grids : all cells and their minimum/maximum latitude/longitude
    '''
    for south, west, north, east, grid_label in final_grids:
        if south <= lat <= north and west <= lng <= east:
            return grid_label
    return None

# Approximate border coordinates of South Korea
south_korea_bounds = [33.10, 124.57, 38.60, 131]

# Read waypoint
directory = dataset_path

for directory_name in os.listdir(directory):
    dir_path=os.path.join(dataset_path, directory_name)
    
    # checking if the directory demo_folder exist or not. s
    labeled_data_directory=os.path.join(labeled_data_path, directory_name)
    
    if not os.path.exists(labeled_data_directory):  
        os.makedirs(labeled_data_directory)
    
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            path_dataframes = []
            path_points = []
            data = pd.read_csv(file_path, encoding='utf-8')

            path_dataframes.append(data)
            points = data[['lat', 'lng']].values.tolist()
            path_points.extend(points)

            # Initialize grid queue
            grid_queue = []
            final_grids = []  # Used to store the final small grid
            initial_lat_step = (south_korea_bounds[2] - south_korea_bounds[0]) / 13
            initial_lon_step = (south_korea_bounds[3] - south_korea_bounds[1]) / 13
            #initial_lat_step *= 0.94
            #initial_lon_step *= 0.94
            
            for i in range(13):
                for j in range(13):
                    south = south_korea_bounds[0] + i * initial_lat_step
                    north = south_korea_bounds[0] + (i + 1) * initial_lat_step
                    west = south_korea_bounds[1] + j * initial_lon_step
                    east = south_korea_bounds[1] + (j + 1) * initial_lon_step
                    grid_queue.append((south, west, north, east, num_to_letter(i) + num_to_letter(j)))

            # Process grid queue
            min_size_km = 0.76  # Minimum grid size (km)
            subdivisions = ['A', 'B', 'C', 'D']  # Split label

            while grid_queue:
                south, west, north, east, grid_label = grid_queue.pop(0)
                grid_size_km = min(geopy.distance.distance((south, west), (south, east)).km,
                                geopy.distance.distance((south, west), (north, west)).km)
                if grid_size_km > min_size_km and is_path_in_grid(south, west, north, east, path_points):
                        mid_lat = (south + north) / 2
                        mid_lon = (west + east) / 2
                        grid_queue.append((south, west, mid_lat, mid_lon, grid_label + 'C'))
                        grid_queue.append((mid_lat, west, north, mid_lon, grid_label + 'A'))
                        grid_queue.append((south, mid_lon, mid_lat, east, grid_label + 'D'))
                        grid_queue.append((mid_lat, mid_lon, north, east, grid_label + 'B'))
                else:
                    final_grids.append((south, west, north, east, grid_label))

            # Assign final grid labels to waypoints
            for data in path_dataframes:
                data['grid_label'] = data.apply(lambda row: get_grid_label(row['lat'], row['lng'], final_grids), axis=1)

            labeled_data_file=os.path.join(labeled_data_directory, filename)

            # Save the updated DataFrame to a new CSV file
            for idx, df in enumerate(path_dataframes):
                df.to_csv(f'{labeled_data_file}.csv', index=False)

