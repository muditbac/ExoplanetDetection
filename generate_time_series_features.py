from tsfresh.feature_extraction.feature_calculators import number_peaks, minimum, maximum, mean, median, length, \
    variance, skewness, kurtosis, sum_values, abs_energy, mean_change, ar_coefficient, \
    percentage_of_reoccurring_datapoints_to_all_datapoints, mean_abs_change, count_below_mean, has_duplicate_min, \
    count_above_mean, has_duplicate_max, standard_deviation, absolute_sum_of_changes, last_location_of_minimum, \
    last_location_of_maximum, first_location_of_maximum, longest_strike_below_mean, longest_strike_above_mean, \
    sum_of_reoccurring_values, first_location_of_minimum, sum_of_reoccurring_data_points, \
    variance_larger_than_standard_deviation, ratio_value_number_to_time_series_length, \
    percentage_of_reoccurring_values_to_all_values, agg_autocorrelation, agg_linear_trend, \
    augmented_dickey_fuller, autocorrelation, binned_entropy, c3, cid_ce, energy_ratio_by_chunks, \
    fft_aggregated, fft_coefficient, friedrich_coefficients, index_mass_quantile, linear_trend, \
    partial_autocorrelation, spkt_welch_density
import pandas as pd
import os

from config import raw_data_filename, FEATURES_PATH
from utils.processing_helper import save_features, make_dir_if_not_exists, features_exists


def generate_time_series_feats(x_dataset, dataset_name="raw"):
    make_dir_if_not_exists(os.path.join(FEATURES_PATH, 'tsfeats'))
    time_length = x_dataset.shape[1]

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

        "binned_entropy_max300": lambda x: binned_entropy(x, 300),
        "binned_entropy_max400": lambda x: binned_entropy(x, 400),
        "cid_ce_true": lambda x: cid_ce(x, True),
        "cid_ce_false": lambda x: cid_ce(x, False),

        "percentage_of_reoccurring_datapoints_to_all_datapoints": percentage_of_reoccurring_datapoints_to_all_datapoints
    }

    for feature_name, function_call in features_function_dict.iteritems():
        print "{:.<70s}".format("- Processing feature: %s" % feature_name),
        feature_name = 'tsfeats/%s_%s' % (dataset_name, feature_name)
        if not features_exists(feature_name):
            feats = x_dataset.apply(function_call, axis=1, raw=True).values
            save_features(feats, feature_name)
            print("Done")
        else:
            print("Already generated")

    ar_param_k100 = [{"coeff": i, "k": 100} for i in range(100+1)]
    ar_param_k500 = [{"coeff": i, "k": 500} for i in range(500+1)]
    agg50_mean_linear_trend = [{"attr": val, "chunk_len": 50, "f_agg": "mean"} for val in ("pvalue", "rvalue", "intercept", "slope", "stderr")]
    aug_dickey_fuler_params = [{"attr": "teststat"}, {"attr": "pvalue"}, {"attr": "usedlag"}]
    energy_ratio_num10_focus5 = [{"num_segments": 10, "segment_focus": 5}]
    fft_aggr_spectrum = [{"aggtype": "centroid"}, {"aggtype": "variance"}, {"aggtype": "skew"}, {"aggtype": "kurtosis"}]
    fft_coefficient_real = [{"coeff": i, "attr": "real"} for i in range((time_length+1)//2)]
    fft_coefficient_imag = [{"coeff": i, "attr": "imag"} for i in range((time_length+1)//2)]
    fft_coefficient_abs = [{"coeff": i, "attr": "abs"} for i in range((time_length+1)//2)]
    fft_coefficient_angle = [{"coeff": i, "attr": "angle"} for i in range((time_length+1)//2)]
    linear_trend_params = [{"attr": val}for val in ("pvalue", "rvalue", "intercept", "slope", "stderr")]

    other_feats_dict = {
        "ar_coeff100": lambda x: dict(ar_coefficient(x, ar_param_k100)),
        "ar_coeff500": lambda x: dict(ar_coefficient(x, ar_param_k500)),
        "agg50_mean_lin_trend": lambda x: dict(agg_linear_trend(x, agg50_mean_linear_trend)),
        "aug_dickey_fuler": lambda x: dict(augmented_dickey_fuller(x, aug_dickey_fuler_params)),
        "energy_ratio_num10_focus5": lambda x: dict(energy_ratio_by_chunks(x, energy_ratio_num10_focus5)),
        "fft_aggr_spectrum": lambda x: dict(fft_aggregated(x, fft_aggr_spectrum)),
        "fft_coeff_real": lambda x: dict(fft_coefficient(x, fft_coefficient_real)),
        "fft_coeff_imag": lambda x: dict(fft_coefficient(x, fft_coefficient_imag)),
        "fft_coeff_abs": lambda x: dict(fft_coefficient(x, fft_coefficient_abs)),
        "fft_coeff_angle": lambda x: dict(fft_coefficient(x, fft_coefficient_angle)),
        "linear_trend": lambda x: dict(linear_trend(x, linear_trend_params)),
    }

    for feature_name, function_call in other_feats_dict.iteritems():
        print "{:.<70s}".format("- Processing features: %s" % feature_name),
        feature_name = 'tsfeats/%s_%s' % (dataset_name, feature_name)
        if not features_exists(feature_name):
            feats_dict = x_dataset.apply(function_call, axis=1, raw=True).values.tolist()
            feats = pd.DataFrame.from_dict(feats_dict)
            save_features(feats.values, feature_name)
            print("Done")
        else:
            print("Already generated")

    # Auto-correlations as features
    # TODO Add corr features

if __name__ == '__main__':
    dataset = pd.read_csv(raw_data_filename)

    print('Extracting time series features...')
    x_dataset = dataset.iloc[:, 1:]
    generate_time_series_feats(x_dataset)
