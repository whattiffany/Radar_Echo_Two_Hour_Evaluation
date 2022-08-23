import numpy as np

def nws_precip_colors():

    nan_zero = [
            "#f0f0f0",  #  nan
            "#ffffff"   #  0.00 
    ]

    nws_precip_colors_original = [        
            "#04e9e7",  #  0.01
            "#019ff4",  #  5.00
            "#0300f4",  # 10.00
            "#02fd02",  # 15.00
            "#01c501",  # 20.00
            "#008e00",  # 25.00
            "#fdf802",  # 30.00
            "#e5bc00",  # 35.00
            "#fd9500",  # 40.00
            "#fd0000",  # 45.00
            "#d40000",  # 50.00
            "#bc0000",  # 55.00
            "#f800fd",  # 60.00
            "#9854c6",  # 65.00
        ]

    color_int = []
    for i, val in enumerate(nws_precip_colors_original[:-1]):
        red = [val for val in np.linspace(int(nws_precip_colors_original[i][1:3], 16), int(nws_precip_colors_original[i+1][1:3], 16), 500)]
        green = [val for val in np.linspace(int(nws_precip_colors_original[i][3:5], 16), int(nws_precip_colors_original[i+1][3:5], 16), 500)]
        blue = [val for val in np.linspace(int(nws_precip_colors_original[i][5:7], 16), int(nws_precip_colors_original[i+1][5:7], 16), 500)]
        stack = np.vstack([red, green, blue]).T
        color_int = stack if color_int == [] else np.concatenate((color_int, stack))
    
    color_code = []
    for val in color_int:
        color_code.append('#{:02X}{:02X}{:02X}'.format(int(val[0]), int(val[1]), int(val[2])))
    color_code = np.concatenate([nan_zero, color_code])
    
    return color_code