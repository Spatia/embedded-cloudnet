def format_params(num):
    if num >= 1_000_000:
        return f"{num / 1_000_000:.0f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}k"
    return str(num)

def build_model_name(depthwise, num_params, resize, dilatation_rates=None):
    return f"unet{'_dw' if depthwise else ''}_{format_params(num_params)}{'_aspp' if dilatation_rates else ''}{f'_ds_{resize[0]}' if resize else ''}.pth"