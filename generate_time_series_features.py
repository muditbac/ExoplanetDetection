from tsfresh.feature_extraction.feature_calculators import number_peaks, minimum, maximum, mean, median, length, \
    variance, skewness, kurtosis, sum_values, abs_energy, mean_change, ar_coefficient, \
    percentage_of_reoccurring_datapoints_to_all_datapoints, mean_abs_change, count_below_mean, has_duplicate_min, \
    count_above_mean, has_duplicate_max, standard_deviation, absolute_sum_of_changes, last_location_of_minimum, \
    last_location_of_maximum, first_location_of_maximum, longest_strike_below_mean, longest_strike_above_mean, \
    sum_of_reoccurring_values, first_location_of_minimum, sum_of_reoccurring_data_points, \
    variance_larger_than_standard_deviation, ratio_value_number_to_time_series_length, \
    percentage_of_reoccurring_values_to_all_values
import pandas as pd

from config import raw_data_filename
from utils.processing_helper import save_features


def generate_time_series_feats(x_dataset):
    features_function_dict = {
        "mean": mean,
        "median": median,
        "length": length,
        "minimum": minimum,
        "maximum": maximum,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "sum_values": sum_values,
        "abs_energy": abs_energy,
        "mean_change": mean_change,
        "mean_abs_change": mean_abs_change,
        "count_below_mean": count_below_mean,
        "count_above_mean": count_above_mean,
        "has_duplicate_min": has_duplicate_min,
        "has_duplicate_max": has_duplicate_max,
        "standard_deviation": standard_deviation,
        "absolute_sum_of_changes": absolute_sum_of_changes,
        "last_location_of_minimum": last_location_of_minimum,
        "last_location_of_maximum": last_location_of_maximum,
        "first_location_of_maximum": first_location_of_maximum,
        "longest_strike_below_mean": longest_strike_below_mean,
        "longest_strike_above_mean": longest_strike_above_mean,
        "sum_of_reoccurring_values": sum_of_reoccurring_values,
        "first_location_of_minimum": first_location_of_minimum,
        "sum_of_reoccurring_data_points": sum_of_reoccurring_data_points,
        "variance_larger_than_standard_deviation": variance_larger_than_standard_deviation,
        "ratio_value_number_to_time_series_length": ratio_value_number_to_time_series_length,
        "percentage_of_reoccurring_values_to_all_values": percentage_of_reoccurring_values_to_all_values,
        "percentage_of_reoccurring_datapoints_to_all_datapoints": percentage_of_reoccurring_datapoints_to_all_datapoints
    }

    for feature_name, function_call in features_function_dict.iteritems():
        print "- Processing feature: %s" % feature_name
        feats = x_dataset.apply(function_call, axis=1, raw=True).values
        save_features(feats, "raw_%s" % feature_name)

    ar_param_k100 = [{"coeff": i, "k": 100} for i in range(100+1)]
    ar_param_k500 = [{"coeff": i, "k": 500} for i in range(500+1)]
    other_feats_dict = {
        "ar_coeff100": lambda x: dict(ar_coefficient(x, ar_param_k100)),
        "ar_coeff500": lambda x: dict(ar_coefficient(x, ar_param_k500))
    }

    for feature_name, function_call in other_feats_dict.iteritems():
        print "- Processing features: %s" % feature_name
        feats_dict = x_dataset.apply(function_call, axis=1, raw=True).values.tolist()
        feats = pd.DataFrame.from_dict(feats_dict)
        save_features(feats.values, "raw_%s" % feature_name)

if __name__ == '__main__':
    dataset = pd.read_csv(raw_data_filename)

    print('Extracting time series features...')
    x_dataset = dataset.iloc[:, 1:]
    generate_time_series_feats(x_dataset)
