from src.data.saml_homo import load_and_prepare_saml_data

def load_and_prepare_data(sample_Ratio = 1.0, use_cache = True):
    return load_and_prepare_saml_data(sample_ratio=sample_Ratio, use_cache=use_cache)