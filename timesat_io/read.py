# External Imports
from typing import List, Optional, Union
import io
import numpy as np


### Helper Classes ###

class SeasParamSet:
    def __init__(self, season_id: np.int32, param_list: List[np.float32]):
        self.season_id = season_id
        self.start_of_season = param_list[0]
        self.end_of_season = param_list[1]
        self.length_of_season = param_list[2]
        self.base_level = param_list[3]
        self.mid_of_season = param_list[4]
        self.largest_data_value = param_list[5]
        self.seasonal_amplitude = param_list[6]
        self.rate_of_increase = param_list[7]
        self.rate_of_decrease = param_list[8]
        self.large_seasonal_integral = param_list[9]
        self.small_seasonal_integral = param_list[10]
        self.value_for_start_of_season = param_list[11]
        self.value_for_end_of_season = param_list[12]

    def as_string(self):
        return """
        season_id: {}, 

        Start of Season: {}, 
        End of Season: {}, 
        Length of Season: {}, 
        Base Level: {}, 
        Middle of Season: {}, 
        Largest Data Value: {}, 
        Seasonal Amplitude: {}, 
        Rate of Increase: {}, 
        Rate of Decrease: {},
        Large Seasonal Integral: {},
        Small Seasonal Integral: {},
        Value for Start of Season: {},
        Value for End of Season: {}
        """.format(
            self.season_id,
            self.start_of_season,
            self.end_of_season,
            self.length_of_season,
            self.base_level,
            self.mid_of_season,
            self.largest_data_value,
            self.seasonal_amplitude,
            self.rate_of_increase,
            self.rate_of_decrease,
            self.large_seasonal_integral,
            self.small_seasonal_integral,
            self.value_for_start_of_season,
            self.value_for_end_of_season
        )


class TSCoreSample:
    def __init__(self, row: np.int32, col: np.int32, n_num: np.int32, seasons: List[SeasParamSet]):
        self.row = row
        self.col = col
        self.n_num = n_num
        self.seasons = seasons

    def as_string(self):
        seasons_string = "".join(["".join(["\n-------\n", str(s.as_string()), "\n-------\n"]) for s in self.seasons])

        return "row: {}, col: {}, n_num: {}, seasons: {}".format(
            self.row,
            self.col,
            self.n_num,
            seasons_string
        )


### Core Functions ###

def get_seas_params(byte_data: bytearray, byte_offset: int, specific_parameter: Optional[int] = None) -> (
        Union[TSCoreSample, np.float32], int):
    # Use a higher-performance memoryview (no copy)
    byte_data = memoryview(byte_data)

    # Grab "header" information using specified bytes
    row = np.frombuffer(buffer=byte_data[byte_offset:byte_offset + 4],
                        dtype=np.uint32)[0]

    col = np.frombuffer(buffer=byte_data[byte_offset + 4:byte_offset + 8],
                        dtype=np.uint32)[0]

    n_seasons = np.frombuffer(buffer=byte_data[byte_offset + 8:byte_offset + 12],
                              dtype=np.uint32)[0]

    # Create a list for the seasonal parameter objects or values to be stored inside of
    seasons_with_params = []

    # Get seasonal parameters for each of the available seasons
    season_offset = None
    for i in range(0, n_seasons):
        # Create a "logical season" that lines up with TIMESAT seasons
        logical_season = i + 1

        # Calculate the offset specific to the next season's params
        # Note: The 12 bit offset jumps ahead of the row/col/n_x information
        # Note: Each new season is 52 bits ahead of the last sequentially
        season_offset = byte_offset + 12 + i * 52

        # Read correct bytes using previously calculated offset
        raw_byte_data = byte_data[season_offset:season_offset + 52]

        # Create a numpy array from the raw bytes
        vals = np.frombuffer(buffer=raw_byte_data,
                             dtype=np.float32)

        # Make sure vals is the correct length
        # Note: The length of the seasonal parameter array should ALWAYS be 13
        assert len(vals) == 13

        # Pick/generate output based on user-defined parameter
        options = {
            None: SeasParamSet(logical_season, vals),  # Create seasonal params object and append
            1: vals[0],
            2: vals[1],
            3: vals[2],
            4: vals[3],
            5: vals[4],
            6: vals[5],
            7: vals[6],
            8: vals[7],
            9: vals[8],
            10: vals[9],
            11: vals[10],
            12: vals[11],
            13: vals[12]
        }
        seasons_with_params.append(options[specific_parameter])

    # The season byte offset should never be None unless there were no seasons!
    assert season_offset is not None

    # Note: 52 is added to the season byte offset to jump over the last season
    output_offset = season_offset + 52

    # Pick value to pack in first location of output tuple
    output_value = None
    if specific_parameter is None:
        output_value = TSCoreSample(row=row, col=col, n_num=n_seasons, seasons=seasons_with_params)
    else:
        output_value = seasons_with_params

    # Build output object
    return {"row": row, "col": col, "vals": output_value, "offset": output_offset}


def get_time_series(byte_data: bytearray, byte_offset: int, num_samples: int) -> np.ndarray:
    # Use a higher-performance memoryview (no copy)
    byte_data = memoryview(byte_data)

    # Grab "header" information using specified bytes
    row = np.frombuffer(buffer=byte_data[byte_offset:byte_offset + 4],
                        dtype=np.uint32)[0]

    col = np.frombuffer(buffer=byte_data[byte_offset + 4:byte_offset + 8],
                        dtype=np.uint32)[0]

    # Data Offset
    data_offset = byte_offset + 8

    # Data End Offset (non-inclusive)
    # Note: Multiply by 4 because 32/8 == 4. Floats are 32-bit, bytes are 8 bits.
    end_offset = data_offset + int(num_samples * 4)

    # Convert byte slice into a numpy array of float32 values
    vals = np.frombuffer(buffer=byte_data[data_offset:end_offset],
                         dtype=np.float32)

    # Return dictionary containing pixel information and time series values
    return {"row": row, "col": col, "vals": vals, "offset": end_offset}


def tpa_read_in(file_path: str, specific_parameter: Optional[int] = None, debug: Optional[bool] = False) -> np.ndarray:
    ### Open and Read In the File ###

    # Open the file using OS
    file = io.open(file_path, mode='rb', buffering=0)

    # Read all the data (no buffering) into a bytes object
    data = file.readall()

    # Store the size of the data (in bytes) for later use
    data_size = len(data)

    if debug:
        print("[DEBUG] Data Type: {}".format(type(data)))
        print("[DEBUG] Size (bytes): {}".format(data_size))

    ### Decode "Header" for File-Wide Information ###

    # The 'nyears' information resides in bytes 0 to 4 (exclusive)
    nyears = np.frombuffer(buffer=data[0:4], dtype=np.uint32)[0]

    # The 'nptperyear' or 'number of points per year' information resides in bytes 4 to 8 (exclusive)
    nptperyear = np.frombuffer(buffer=data[4:8], dtype=np.uint32)[0]

    # The 'rowstart' information resides in bytes 8 to 12 (exclusive)
    rowstart = np.frombuffer(buffer=data[8:12], dtype=np.uint32)[0]

    # The 'rowstop' information resides in bytes 12 to 16 (exclusive)
    rowstop = np.frombuffer(buffer=data[12:16], dtype=np.uint32)[0]

    # The 'colstart' information resides in bytes 16 to 20 (exclusive)
    colstart = np.frombuffer(buffer=data[16:20], dtype=np.uint32)[0]

    # The 'colstop' information resides in bytes 20 to 24 (exclusive)
    colstop = np.frombuffer(buffer=data[20:24], dtype=np.uint32)[0]

    ### Read Data for Each Season, for Each Pixel ###

    # Come up with a 'total number of seasons' value
    # [FIXME] Number of seasons is very brittle!
    num_of_seasons = nyears - 1

    # Create a numpy array to store the data in
    pixel_array = np.ndarray((num_of_seasons, rowstop, colstop), dtype=np.float32)
    pixel_array.fill(np.float32(0))

    if debug:
        print("[DEBUG] Created data array:\n\tSeasons: {}\n\tRows: {}\n\tColumns: {}\n".format(
            len(pixel_array), len(pixel_array[0]), len(pixel_array[0][0])))

    # Run the function to get all the seasonal parameters for each pixel
    next_offset = 24  # First 24 bits are occupied by the "header"
    while next_offset < data_size:
        # Call function that gets the seasonal parameters for an unknown pixel (that hasn't been seen yet)
        output = get_seas_params(byte_data=data, byte_offset=next_offset, specific_parameter=specific_parameter)

        # Write samples to the given "pixel" without an explicit loop
        pixel_array[:, output["row"] - 1, output["col"] - 1] = output["vals"]

        # Reset the next offset based on the value calculated by the get_seas_params function
        next_offset = output["offset"]

    ### Return the Value ###
    return pixel_array

def tts_read_in(file_path: str, debug: Optional[bool] = False):
    ### Open and Read In a .tts Time Series File ###

    # Open the file using OS
    with io.open(file_path, mode='rb', buffering=0) as file:
        # Read all the data (no buffering) into a bytes object
        data = file.readall()

    # Store the size of the data (in bytes) for later use
    data_size = len(data)

    if debug:
        print("[DEBUG] Data Type: {}".format(type(data)))
        print("[DEBUG] Size (bytes): {}".format(data_size))

    ### Decode "Header" for File-Wide Information ###

    # The 'nyears' information resides in bytes 0 to 4 (exclusive)
    nyears = np.frombuffer(buffer=data[0:4], dtype=np.uint32)[0]

    # The 'nptperyear' or 'number of points per year' information resides in bytes 4 to 8 (exclusive)
    nptperyear = np.frombuffer(buffer=data[4:8], dtype=np.uint32)[0]

    # The 'rowstart' information resides in bytes 8 to 12 (exclusive)
    rowstart = np.frombuffer(buffer=data[8:12], dtype=np.uint32)[0]

    # The 'rowstop' information resides in bytes 12 to 16 (exclusive)
    rowstop = np.frombuffer(buffer=data[12:16], dtype=np.uint32)[0]

    # The 'colstart' information resides in bytes 16 to 20 (exclusive)
    colstart = np.frombuffer(buffer=data[16:20], dtype=np.uint32)[0]

    # The 'colstop' information resides in bytes 20 to 24 (exclusive)
    colstop = np.frombuffer(buffer=data[20:24], dtype=np.uint32)[0]

    ### Read Time Series Data for Each Pixel ###

    # Come up with a 'total number of samples' value
    total_samples = nyears * nptperyear

    # Create a numpy array to store the data in
    pixel_array = np.ndarray((total_samples, rowstop, colstop), dtype=np.float32)
    pixel_array.fill(np.float32(0))

    if debug:
        print("[DEBUG] Created data array:\n\tSamples: {}\n\tRows: {}\n\tColumns: {}\n".format(
            len(pixel_array), len(pixel_array[0]), len(pixel_array[0][0])))

    # Run the function to get a time series (1D array) for each pixel
    next_offset = 24
    while next_offset < data_size:
        # Run function to get all the time series samples for the given pixel at the offset
        output = get_time_series(byte_data=data, byte_offset=next_offset, num_samples=total_samples)

        # Write samples to the given "pixel" without an explicit loop
        pixel_array[:, output["row"] - 1, output["col"] - 1] = output["vals"]

        # Reset the next offset based on the value calculated by the get_seas_params function
        next_offset = output["offset"]

    ### Return the Value ###
    return pixel_array
