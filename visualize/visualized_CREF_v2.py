import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from visualize.colormap import nws_precip_colors
from area_20 import area_20
shape_path = 'visualize/mapdata201805310314/COUNTY_MOI_1070516'

""" 
    visualized_cloud Parameter:

    將二維雷達回波資料視覺化

    data      輸入二維資料，不限制輸入矩陣尺吋
    title     圖表顯示標題


    visualized_with_map Parameter:

    將二維雷達回波資料視覺化，並加入縣市線格

    data      輸入二維資料，輸入矩陣尺吋為 (881, 921)、 (561, 441)。
    title     圖表顯示標題
    savepath  圖表儲存路徑，有給路徑才會進行儲存
    hiroi     因應雷達回波兩種範圍格式，True為(881, 921) | False為(561, 441)。
    terrain   是否顯示地型樣貌

"""


def visualized_cloud(data, title=None):
    y, x = data.shape
    print(y)
    print(x)
    x, y = int(str(x)[0]), int(str(y)[0])
    print(int(str(x)[0]))
    print(int(str(y)[0]))
    fig, ax = plt.subplots()

    if title:
        ax.title.set_text(title)

    precip_colormap = colors.ListedColormap(nws_precip_colors())
    
    # mapping color bar
    levels = np.linspace(0, 65, 14)
    levels2 = np.concatenate((np.array([-999., -99., 0.01]), np.linspace(1, 65, len(nws_precip_colors())-101)))
    norm = colors.BoundaryNorm(boundaries=levels2, ncolors=levels2.shape[0]-1)
    im = ax.pcolormesh(data, norm=norm, cmap=precip_colormap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    plt.colorbar(im, ax=ax, cax=cax, ticks=levels)
    
    return fig, ax


def visualized_with_map(radar, title='', savepath=None, hiroi=False, terrain=False):
    if hiroi:
        m = Basemap(projection='merc', resolution='i', fix_aspect=True,
                    llcrnrlon=115.0, llcrnrlat=18.0,
                    urcrnrlon=126.5125, urcrnrlat=29.0125,
                    lat_ts=20)
        m.drawparallels(np.arange(18, 29.0125), fontsize=10)
        m.drawmeridians(np.arange(115, 126.5125), fontsize=10)
        lons = np.linspace(115, 126.5125, 1501)
        lats = np.linspace(18, 29.0125, 1501)
    
    else:
        m = Basemap(projection='merc', resolution='i', fix_aspect=True,
                      llcrnrlon=118.0, llcrnrlat=19.9875,
                      urcrnrlon=123.5125, urcrnrlat=27.0,
                      lat_ts=20)
        m.drawparallels(np.arange(19, 27), fontsize=10)
        m.drawmeridians(np.arange(118, 124), fontsize=10)
        lons = np.linspace(118, 123.5125, 441)
        lats = np.linspace(19.9875, 27, 561)

    llons, llats = np.meshgrid(lons, lats)
    x, y = m(llons, llats)

    m.drawcoastlines()

    # terrain
    if terrain:
        m.etopo()

    m.readshapefile(shape_path , linewidth=0.25 , drawbounds=True, name='Taiwan')

    precip_colormap = colors.ListedColormap(nws_precip_colors())

    # mapping color bar
    levels = np.linspace(0, 65, 14)
    levels2 = np.concatenate((np.array([-999., -99., 0.01]), np.linspace(1, 65, len(nws_precip_colors())-101)))
    
    norm = colors.BoundaryNorm(boundaries=levels2, ncolors=levels2.shape[0]-1)

    cax = m.pcolormesh(x, y, radar, norm=norm, cmap=precip_colormap)
    m.colorbar(cax, ticks=levels)

    plt.title(str(title))
    plt.xlabel('lon' , fontsize=12 , x=1)
    plt.ylabel('lat' , fontsize=12 , y=1)
    
    if savepath:
        plt.savefig('{}/CREF_AllTaiwan{}.png'.format(savepath, str(title)), dpi=200)

    return m

def visualized_area_with_map2(radar, place, shape_size=[105, 105], title='', savepath=None, terrain=False):
    
    llcrnrlat = area_20[place].lat - 0.01*(shape_size[0]//2)
    urcrnrlat = area_20[place].lat + 0.01*(shape_size[0]//2)
    llcrnrlon = area_20[place].lon - 0.01*(shape_size[1]//2)
    urcrnrlon = area_20[place].lon + 0.01*(shape_size[1]//2)

    x, y = get_xy_hiroi(place)

    radar = radar[y-int(shape_size[0]//2):y+int(shape_size[0]//2)+1, x-int(shape_size[1]//2):x+int(shape_size[1]//2)+1]

    m = Basemap(projection='merc', resolution='i', fix_aspect=True,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                lat_ts=20)

    point_x, point_y = m((llcrnrlon+urcrnrlon)/2, (llcrnrlat+urcrnrlat)/2)
#    m.plot(point_x, point_y, linestyle='none', marker="o", markersize=45, alpha=0.4, c='white', markeredgecolor="red", markeredgewidth=1)
#    m.plot(point_x, point_y, linestyle='none', marker=".", markersize=1, alpha=1, c='red', markeredgecolor="red", markeredgewidth=1)
    point_x, point_y = m((llcrnrlon+urcrnrlon)/2 - 0.01*5, (llcrnrlat+urcrnrlat)/2 + 0.01*11)
#    plt.text(point_x, point_y, place)

    m.drawparallels(np.arange(llcrnrlon, llcrnrlat), fontsize=10)
    m.drawmeridians(np.arange(urcrnrlon, urcrnrlat), fontsize=10)

    lons = np.linspace(llcrnrlon, urcrnrlon, radar.shape[0])
    lats = np.linspace(llcrnrlat, urcrnrlat, radar.shape[1])
  
    llons, llats = np.meshgrid(lons, lats)
    x, y = m(llons, llats)

    # terrain
    if terrain:
        m.etopo()

    m.readshapefile(shape_path , linewidth=0.25 , drawbounds=True, name='Taiwan')
    precip_colormap = colors.ListedColormap(nws_precip_colors())

    # mapping color bar
    levels = np.linspace(0, 65, 14)
    levels2 = np.concatenate((np.array([-999., -99., 0.01]), np.linspace(1, 65, len(nws_precip_colors())-101)))
    
    norm = colors.BoundaryNorm(boundaries=levels2, ncolors=levels2.shape[0]-1)

    cax = m.pcolormesh(x, y, radar, norm=norm, cmap=precip_colormap)
    m.colorbar(cax, ticks=levels)

    plt.title(str(title))
    plt.xlabel('lon' , fontsize=12 , x=1)
    plt.ylabel('lat' , fontsize=12 , y=1)
    
    if savepath:
        plt.savefig('{}/CREF{}.png'.format(savepath, str(title)), dpi=200)

    return m

def get_xy(place=None):
    
    lat = area_20[place].lat
    lon = area_20[place].lon
    x = int(np.ceil((lon - 113.5)/0.01))
    y = int(1501 - np.ceil((31 - lat)/0.01))

    return x, y

def get_xy_hiroi(place=None):
    
#    lat = area[place].lat
#    lon = area[place].lon
#    x = int(np.ceil((lon - 115.0)/0.01))
#    y = int(881 - np.ceil((29.0125 - lat)/0.01))
    lat = area_20[place].lat
    lon = area_20[place].lon
    x = int(np.ceil((lon - 113.5)/0.01))
    y = int(1501 - np.ceil((31 - lat)/0.01))

    return x, y

if __name__ == "__main__":
    
    import gzip, struct
    
    save_path = 'Result/ConvLSTM_nokmean_radarshow_alltaiwa_1111/CREF_20180824_0000pred6/'
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
#    gz = gzip.open("NWP/COMPREF_OBS_FCST/COMPREF_OBS_FCST/CREF_010min/CREF_010min.20180824.000000.gz").read()
    gz = gzip.open("CREF/CREF_010min.20180824.000000.gz").read()
    radar = np.array(struct.unpack(1501*1501*'h', gz[-1501*1501*2:])).reshape(1501, 1501).astype(np.float64)/10
    print(radar)
    print(np.array(radar).shape)
    visualized_area_with_map(radar, 'Sun_Moon_Lake', [393, 393], title='20180824.0010', savepath=save_path)
    
    gz = gzip.open("CREF/CREF_020min.20180824.000000.gz").read()
    radar = np.array(struct.unpack(1501*1501*'h', gz[-1501*1501*2:])).reshape(1501, 1501).astype(np.float64)/10
    print(radar)
    print(np.array(radar).shape)
    visualized_area_with_map(radar, 'Sun_Moon_Lake', [393, 393], title='20180824.0020', savepath=save_path)
    
    gz = gzip.open("CREF/CREF_030min.20180824.000000.gz").read()
    radar = np.array(struct.unpack(1501*1501*'h', gz[-1501*1501*2:])).reshape(1501, 1501).astype(np.float64)/10
    print(radar)
    print(np.array(radar).shape)
    visualized_area_with_map(radar, 'Sun_Moon_Lake', [393, 393], title='20180824.0030', savepath=save_path)
    
    gz = gzip.open("CREF/CREF_040min.20180824.000000.gz").read()
    radar = np.array(struct.unpack(1501*1501*'h', gz[-1501*1501*2:])).reshape(1501, 1501).astype(np.float64)/10
    print(radar)
    print(np.array(radar).shape)
    visualized_area_with_map(radar, 'Sun_Moon_Lake', [393, 393], title='20180824.0040', savepath=save_path)
    
    gz = gzip.open("CREF/CREF_050min.20180824.000000.gz").read()
    radar = np.array(struct.unpack(1501*1501*'h', gz[-1501*1501*2:])).reshape(1501, 1501).astype(np.float64)/10
    print(radar)
    print(np.array(radar).shape)
    visualized_area_with_map(radar, 'Sun_Moon_Lake', [393, 393], title='20180824.0050', savepath=save_path)
    
    gz = gzip.open("CREF/CREF_060min.20180824.000000.gz").read()
    radar = np.array(struct.unpack(1501*1501*'h', gz[-1501*1501*2:])).reshape(1501, 1501).astype(np.float64)/10
    print(radar)
    print(np.array(radar).shape)
    visualized_area_with_map(radar, 'Sun_Moon_Lake', [393, 393], title='20180824.0100', savepath=save_path)
    

   