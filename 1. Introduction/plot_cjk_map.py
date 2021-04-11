import argparse
import glob
import itertools
import json
from pathlib import Path
import re
import string

import pandas as pd
import numpy as np
import scipy as sp

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.path as mpath

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import geojson
import geopandas as gpd
import topojson as tp
import imageio

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from matplotlib.patches import Circle, Ellipse, Rectangle

from pytz import timezone

SCRIPT_DIR = Path(__file__).parent.absolute()
EPSILON = 1e-10

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"

def plot():
    output_dir = SCRIPT_DIR / 'out'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    data_dir = SCRIPT_DIR / 'data'

    cjk_fname = 'countries.geojson'
    seoul_fname = 'seoul_municipalities_geo_simple.json'

    cjk_df = gpd.GeoDataFrame.from_file(SCRIPT_DIR / 'data' / cjk_fname)
    seoul_df = gpd.GeoDataFrame.from_file(SCRIPT_DIR / 'data' / seoul_fname)

    stations_latlon = {
        "중구" : [126.9747, 37.5643],
        "종로구" : [127.0050, 37.5720],
        "용산구" : [127.0048, 37.5400],
        "광진구" : [127.0925, 37.5472],
        "성동구" : [127.0419, 37.5432],
        "중랑구" : [127.0940, 37.5849],
        "동대문구" : [127.0289, 37.5758],
        "성북구" : [127.0273, 37.6067],
        "도봉구" : [127.0290, 37.6542],
        "은평구" : [126.9348, 37.6098],
        "서대문구" : [126.9378, 37.5767],
        "마포구" : [126.9456, 37.5498],
        "강서구" : [126.8351, 37.5447],
        "구로구" : [126.8897, 37.4985],
        "영등포구" : [126.8974, 37.5250],
        "동작구" : [126.9715, 37.4809],
        "관악구" : [126.9271, 37.4874],
        "강남구" : [127.0481, 37.5176],
        "서초구" : [126.9945, 37.5046],
        "송파구" : [127.1165, 37.5218],
        "강동구" : [127.1368, 37.5450],
        "금천구" : [126.9083, 37.4524],
        "강북구" : [127.0288, 37.6379],
        "양천구" : [126.8587, 37.5234],
        "노원구" : [127.0679, 37.6574]}


    seoul_lat = [126.83, 127.09]
    seoul_lon = [37.5, 37,57]

    ax = cjk_df.plot(figsize=(7.22, 6.22), alpha=0.8, color='#fff',
                     edgecolor='#777', facecolor='#add8e6')
    ax.set_facecolor('#add8e6')
    fig = plt.gcf()    
    ax.set_xlim((116, 132))
    ax.set_ylim((32, 45))
    ax.set_aspect(1.0)

    seoul_sbox = (seoul_lat[0], seoul_lon[0])
    seoul_lbox = (117, 38)
    large_box = 6
    small_box = 0.3

    rect = Rectangle((seoul_lat[0], seoul_lon[0]),
                      small_box, small_box,
                      linewidth=0.5, edgecolor='k', facecolor='white', zorder=6)
    ax.add_artist(rect)
    axin_seoul = ax.inset_axes(bounds=[seoul_lbox[0], seoul_lbox[1], large_box, large_box],
        transform=ax.transData, alpha=0.4, zorder=6)

    seoul_df.plot(ax=axin_seoul, color='none',
                  edgecolor='#333', facecolor='none', alpha=0.3)

    point_r = 0.012
    aspect = axin_seoul.get_aspect()
    for station, loc in stations_latlon.items():
        lat, lon = loc[0], loc[1]
        p = Ellipse((lat, lon), point_r, point_r / aspect, fc='#1A4E66', zorder=7)
        # if station == '강서구' or station == '서초구':
        #     p = Ellipse((lat, lon), 2.0 * point_r, 2.0 * (point_r / aspect),
        #                 fc='#E26C22', zorder=7)
        if station == '종로구':
            p = Ellipse((lat, lon), 2.0 * point_r, 2.0 * (point_r / aspect),
                        fc='#00A1F1', zorder=7)
        axin_seoul.add_artist(p)

    # connect rect to inset
    axis_to_data = ax.transAxes + ax.transData.inverted()
    x0, y0 = (seoul_lbox[0], seoul_lbox[1] + 0.5)
    x1, y1 = (seoul_lbox[0] + large_box, seoul_lbox[1] + large_box - 0.5)
    px0, py0 = (seoul_sbox[0], seoul_sbox[1])
    px1, py1 = (seoul_sbox[0] + small_box, seoul_sbox[1] + small_box)
    verts_0 = [(px0, py0), (x0, y0), (px0, py0)]
    verts_1 = [(px1, py1), (x1, y1), (px1, py1)]
    codes_0 = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    codes_1 = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    
    path_0 = mpath.Path(verts_0, codes_0)
    path_1 = mpath.Path(verts_1, codes_1)

    patch_1 = ax.add_patch(mpatches.PathPatch(path_0, facecolor='k', lw=0.5))
    patch_2 = ax.add_patch(mpatches.PathPatch(path_1, facecolor='k', lw=0.5))

    # Hide Axis
    sns.despine(left=True, bottom=True)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    axin_seoul.xaxis.set_visible(False)
    axin_seoul.yaxis.set_visible(False)
    # print(seoul_lbox, large_box)

    plt.tight_layout()
    output_to_plot = "cjk_map"
    png_path = output_dir / (output_to_plot + '.png')
    svg_path = output_dir / (output_to_plot + '.svg')
    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

if __name__ == '__main__':
    plot()
