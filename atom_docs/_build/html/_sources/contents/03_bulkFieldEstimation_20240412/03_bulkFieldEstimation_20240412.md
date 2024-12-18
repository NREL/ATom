```python
%load_ext autoreload
%autoreload 2

# hardware
import os
# analysis
import numpy as np
import xarray as xr
import pandas as pd
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
# vis
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-deep')

# Acoustic tomography package
import atom
# configuration object
from hydra import initialize, compose
from hydra.utils import instantiate
with initialize(version_base=None, config_path="../../conf/"):
    cfg = compose(config_name="configs",)
```

# The `atom.backgroundFlow` module 
Bulk field estimation is done through the Background Flow module.

The first step in estimating the flow fields with `atom` is to find the linear system that describes the bulk flow, which is a spatial average considering the full array domain. The bulk flow estimates are used as an estimate of the the adevection velocity for covariance field calculations and fluctuating field estimates in subsequent steps.

```{note}
Future applications of AT target 3D flows in the ABL, so we need some mechanism to account for non-homogeneous temperature and velocity profiles. These will be developed in future installments to the `atom.backgroundFlow` module.
```


```python
## Load data
### Array data
atarray = instantiate(cfg.atarray)

### Auxiliary data
auxdata = instantiate(cfg.auxdata)
auxDataPath = "/Users/nhamilt2/Documents/ATom/data/Data_collection_20190815/20190815123732_AcouTomAuxData.txt"
auxdata.loadData(auxDataPath)

### Constants
constants = instantiate(cfg.constants)

### TravelTimeExtractor
ttextractor = atom.signalProc.TravelTimeExtractor.from_netcdf('../extractedTravelTimes_example.nc')
```

# Linear System object
This object assembles linear system blocks to estimate the bulk flow within the acoustic tomography array. 

It requires:
   - array geometry from the atArray object
   - (filtered) measured travel times from the travelTimeExtractor object

This class is responsible for assembling all the necessary information for the linear system solution as part of the AT process.

$Gf = b$

$G$ is the geometry block (which can also include direct measurements).

$f = [1/c_0, u_0/c_0^2, v0/c_0^2]$ is the vector of unknowns.

$b$ are observed travel times from travelTimeExtractor

The background flow if found through the solution:

$f = (G^{-1} * G)^T (G^{-1} * b)$

- $G = [I, 3] (I==nMics*nSpeakers)$
- $f = [3]$ (see above)
- $b = I$ (travel times for a frame)


```python
ls = atom.backgroundFlow.linearsystem.LinearSystem(
    atarray=atarray.ds,
    measuredTravelTime=ttextractor.ds.filteredMeasuredTravelTimes,
    constants=constants
)

ls.executeProcess()
# update pathID index to match the definition in the atarray object
ls.ds['pathID'] = atarray.ds.pathID
# store linearSystem dataset as netcdf file
ls.to_netcdf('../bulkField_example_data.nc')
```

The `linearSystem.executeProcess` executes all methods of the LinearSystem in order. This includes building the process block, collecting the observation block, solving the system, and extracting the bulk values.
- `linearSystem.buildProcessBlock()`
- `linearSystem.collectObservationBlock()`
- `linearSystem.solve()`
- `linearSystem.extractBulkValues()`


```python
ls.ds
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:             (frame: 120, pathID: 56, variable: 3, component: 3)
Coordinates:
  * frame               (frame) int64 0 1 2 3 4 5 6 ... 114 115 116 117 118 119
  * pathID              (pathID) object MultiIndex
  * spk                 (pathID) int64 0 0 0 0 0 0 0 1 1 1 ... 6 6 7 7 7 7 7 7 7
  * mic                 (pathID) int64 1 2 3 4 5 6 7 0 2 3 ... 5 7 0 1 2 3 4 5 6
  * variable            (variable) object &#x27;ones&#x27; &#x27;ncos&#x27; &#x27;nsin&#x27;
  * component           (component) int64 0 1 2
Data variables:
    measuredTravelTime  (frame, pathID) float64 0.09498 0.136 ... 0.1961 0.1233
    pathOrientation     (pathID) float64 -0.3769 -0.9134 ... -1.671 -1.935
    pathLength          (pathID) float64 32.52 46.28 76.65 ... 86.81 67.15 41.81
    arrayGeom           (pathID, variable) float64 1.0 -0.9298 ... 0.3562 0.9344
    observationalData   (pathID, frame) float64 0.002921 0.002921 ... 0.002948
    variableBlock       (component, frame) float64 0.002929 ... 9.648e-06
    c                   (frame) float64 341.4 341.2 341.0 ... 342.0 341.8 341.5
    u                   (frame) float64 4.98 4.678 5.111 ... 5.815 5.714 6.154
    v                   (frame) float64 1.815 1.573 1.73 ... 1.151 1.201 1.125
    T                   (frame) float64 290.1 289.7 289.4 ... 291.1 290.6 290.2</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-4ab19e07-9170-4505-b4f8-7cf61f52d46f' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-4ab19e07-9170-4505-b4f8-7cf61f52d46f' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>frame</span>: 120</li><li><span class='xr-has-index'>pathID</span>: 56</li><li><span class='xr-has-index'>variable</span>: 3</li><li><span class='xr-has-index'>component</span>: 3</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-86054bc6-2b5f-40c4-a2cd-3271c62d7531' class='xr-section-summary-in' type='checkbox'  checked><label for='section-86054bc6-2b5f-40c4-a2cd-3271c62d7531' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>frame</span></div><div class='xr-var-dims'>(frame)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-d4d00e9e-7863-4c39-98ad-45adc022328c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d4d00e9e-7863-4c39-98ad-45adc022328c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-35edcbe2-f0eb-4a44-b469-72dffffd3f90' class='xr-var-data-in' type='checkbox'><label for='data-35edcbe2-f0eb-4a44-b469-72dffffd3f90' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>pathID</span></div><div class='xr-var-dims'>(pathID)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>MultiIndex</div><input id='attrs-97935c50-7d1d-4d2d-8ceb-b52a7211d0ad' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-97935c50-7d1d-4d2d-8ceb-b52a7211d0ad' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f36d315b-6d1d-4a1f-845c-9245d5d2349a' class='xr-var-data-in' type='checkbox'><label for='data-f36d315b-6d1d-4a1f-845c-9245d5d2349a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 2),
       (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 3), (2, 4),
       (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6),
       (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (5, 0),
       (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2),
       (6, 3), (6, 4), (6, 5), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),
       (7, 5), (7, 6)], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>spk</span></div><div class='xr-var-dims'>(pathID)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 1 ... 6 7 7 7 7 7 7 7</div><input id='attrs-c19a283c-2bae-4304-bf26-f544c37e8039' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c19a283c-2bae-4304-bf26-f544c37e8039' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-de137b5c-273b-49e5-87ee-7f4008d637aa' class='xr-var-data-in' type='checkbox'><label for='data-de137b5c-273b-49e5-87ee-7f4008d637aa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
       3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
       6, 7, 7, 7, 7, 7, 7, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>mic</span></div><div class='xr-var-dims'>(pathID)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1 2 3 4 5 6 7 0 ... 7 0 1 2 3 4 5 6</div><input id='attrs-a56a00b5-96a7-4afb-ab85-8cac44c5b0eb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a56a00b5-96a7-4afb-ab85-8cac44c5b0eb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-529689d0-85bb-4605-8b82-904bc21399cd' class='xr-var-data-in' type='checkbox'><label for='data-529689d0-85bb-4605-8b82-904bc21399cd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 7, 0, 1, 3, 4, 5, 6, 7, 0, 1, 2,
       4, 5, 6, 7, 0, 1, 2, 3, 5, 6, 7, 0, 1, 2, 3, 4, 6, 7, 0, 1, 2, 3, 4, 5,
       7, 0, 1, 2, 3, 4, 5, 6])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>variable</span></div><div class='xr-var-dims'>(variable)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;ones&#x27; &#x27;ncos&#x27; &#x27;nsin&#x27;</div><input id='attrs-7584b0c4-8c8f-40ae-b864-75c060a744fb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7584b0c4-8c8f-40ae-b864-75c060a744fb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7b983a2b-9bc0-4f26-a274-999e2838c2bd' class='xr-var-data-in' type='checkbox'><label for='data-7b983a2b-9bc0-4f26-a274-999e2838c2bd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;ones&#x27;, &#x27;ncos&#x27;, &#x27;nsin&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>component</span></div><div class='xr-var-dims'>(component)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2</div><input id='attrs-7be8d07c-fa13-40e1-ad1f-411ca34b9b6e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7be8d07c-fa13-40e1-ad1f-411ca34b9b6e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-590cdfe4-7636-4a46-86d2-9b0c8f6abcf5' class='xr-var-data-in' type='checkbox'><label for='data-590cdfe4-7636-4a46-86d2-9b0c8f6abcf5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-45c11bee-2586-480e-bec6-a4589e3d6ce2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-45c11bee-2586-480e-bec6-a4589e3d6ce2' class='xr-section-summary' >Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>measuredTravelTime</span></div><div class='xr-var-dims'>(frame, pathID)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.09498 0.136 ... 0.1961 0.1233</div><input id='attrs-39eade07-189b-4368-ade2-0207053a5df2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-39eade07-189b-4368-ade2-0207053a5df2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-77cc4625-85b7-4959-a4a6-3e5c1399337f' class='xr-var-data-in' type='checkbox'><label for='data-77cc4625-85b7-4959-a4a6-3e5c1399337f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.09497939, 0.13602212, 0.21786678, ..., 0.2536304 , 0.19662443,
        0.1244984 ],
       [0.09500378, 0.13599899, 0.22184115, ..., 0.25369161, 0.19703371,
        0.12449591],
       [0.09500726, 0.1360433 , 0.22462741, ..., 0.25364203, 0.19723209,
        0.12470049],
       ...,
       [0.09397933, 0.13614229, 0.22399569, ..., 0.25156798, 0.19607251,
        0.12325095],
       [0.09393222, 0.13605575, 0.2230483 , ..., 0.25153884, 0.19608419,
        0.12321423],
       [0.09385748, 0.13604142, 0.22398469, ..., 0.25152955, 0.19611589,
        0.1232609 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>pathOrientation</span></div><div class='xr-var-dims'>(pathID)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.3769 -0.9134 ... -1.671 -1.935</div><input id='attrs-97397e07-4a98-4ce5-94d2-9eaf328c28a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-97397e07-4a98-4ce5-94d2-9eaf328c28a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8f0b28ac-1ab9-4692-9e49-4002dd625911' class='xr-var-data-in' type='checkbox'><label for='data-8f0b28ac-1ab9-4692-9e49-4002dd625911' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-0.37691463, -0.91343543, -1.41618909, -1.71492581, -2.20765121,
       -2.55506238,  3.0975226 ,  2.7857375 , -1.64013764, -1.84652822,
       -2.11608861, -2.53812981, -2.84712393,  2.95907326,  2.23403859,
        1.49462119, -1.96597625, -2.30873468, -2.77617808, -3.12273497,
        2.64317197,  1.72733873,  1.2979687 ,  1.18421425, -2.95845059,
        2.97524033,  2.63854848,  2.17544954,  1.42710925,  1.02699164,
        0.83626655,  0.18683571,  2.76538494,  2.39098429,  1.92232046,
        0.93338113,  0.60389255,  0.36650961, -0.16362753, -0.36925948,
        1.86442666,  1.46804943,  0.58484787,  0.2936263 ,  0.01862048,
       -0.5019562 , -0.74795648, -1.27371074,  1.19837214, -0.04485248,
       -0.18587245, -0.50005045, -0.96619503, -1.21811826, -1.67065615,
       -1.93499452])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>pathLength</span></div><div class='xr-var-dims'>(pathID)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>32.52 46.28 76.65 ... 67.15 41.81</div><input id='attrs-4064ce7c-e746-47c0-92c9-efa43fd265e3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4064ce7c-e746-47c0-92c9-efa43fd265e3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9c77d834-518a-49b3-84a3-652b38e848e9' class='xr-var-data-in' type='checkbox'><label for='data-9c77d834-518a-49b3-84a3-652b38e848e9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([32.51973447, 46.27924491, 76.65331393, 80.76947335, 81.18959316,
       67.81522334, 40.97077966, 31.39279288, 25.40204482, 66.97489665,
       80.28064128, 95.11943104, 90.41059098, 72.16774045, 45.14719695,
       24.03369587, 43.13015711, 59.46958956, 82.11027343, 84.90409603,
       78.94335669, 76.04688307, 65.9440853 , 41.93356676, 24.47836851,
       61.59935877, 78.69820964, 93.93347136, 80.50664018, 79.53875555,
       58.47771231, 23.12240094, 40.17462912, 62.30906214, 87.16950238,
       81.53310992, 94.93866193, 81.60809622, 60.5224064 , 38.93330432,
       29.61347256, 68.04385105, 68.57432094, 90.68318323, 84.8577106 ,
       77.98955531, 61.33571052, 28.29865963, 42.96536228, 42.24148242,
       73.08282296, 79.61293745, 93.92645558, 86.81220051, 67.14550884,
       41.81356571])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>arrayGeom</span></div><div class='xr-var-dims'>(pathID, variable)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.0 -0.9298 ... 0.3562 0.9344</div><input id='attrs-f1d488b8-3d0d-42e1-8cc6-6a2c01a5f758' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-f1d488b8-3d0d-42e1-8cc6-6a2c01a5f758' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b5e4d806-6494-4b91-a675-d664862cc1b3' class='xr-var-data-in' type='checkbox'><label for='data-b5e4d806-6494-4b91-a675-d664862cc1b3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Array geometry block for linear system</dd><dt><span>units :</span></dt><dd>[- radians radians]</dd></dl></div><div class='xr-var-data'><pre>array([[ 1.        , -0.92980464,  0.36805344],
       [ 1.        , -0.61102985,  0.79160756],
       [ 1.        , -0.15399204,  0.98807209],
       [ 1.        ,  0.143631  ,  0.98963131],
       [ 1.        ,  0.59466981,  0.80397003],
       [ 1.        ,  0.83286609,  0.55347455],
       [ 1.        ,  0.99902907, -0.04405579],
       [ 1.        ,  0.9373489 , -0.34839207],
       [ 1.        ,  0.06928576,  0.99759685],
       [ 1.        ,  0.27225126,  0.9622262 ],
       [ 1.        ,  0.518668  ,  0.85497573],
       [ 1.        ,  0.8233754 ,  0.56749709],
       [ 1.        ,  0.95695647,  0.29023148],
       [ 1.        ,  0.98338952, -0.18150769],
       [ 1.        ,  0.61567499, -0.7880002 ],
       [ 1.        , -0.07610149, -0.99710008],
       [ 1.        ,  0.38497425,  0.92292731],
       [ 1.        ,  0.67276402,  0.73985713],
       [ 1.        ,  0.9339757 ,  0.35733653],
       [ 1.        ,  0.9998222 ,  0.01885657],
...
       [ 1.        , -0.82313147, -0.56785085],
       [ 1.        , -0.93358384, -0.35835905],
       [ 1.        , -0.98664286,  0.16289835],
       [ 1.        , -0.93259487,  0.36092493],
       [ 1.        ,  0.28942908, -0.95719946],
       [ 1.        , -0.10256621, -0.99472618],
       [ 1.        , -0.83379608, -0.55207255],
       [ 1.        , -0.95720063, -0.28942522],
       [ 1.        , -0.99982664, -0.0186194 ],
       [ 1.        , -0.87664303,  0.48114135],
       [ 1.        , -0.73308028,  0.68014212],
       [ 1.        , -0.29273471,  0.9561937 ],
       [ 1.        , -0.36387451, -0.93144798],
       [ 1.        , -0.9989943 ,  0.04483744],
       [ 1.        , -0.98277539,  0.18480403],
       [ 1.        , -0.87755837,  0.47946981],
       [ 1.        , -0.5684341 ,  0.8227288 ],
       [ 1.        , -0.34541228,  0.93845104],
       [ 1.        ,  0.09969394,  0.99501815],
       [ 1.        ,  0.35620019,  0.93440967]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>observationalData</span></div><div class='xr-var-dims'>(pathID, frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.002921 0.002921 ... 0.002948</div><input id='attrs-a2a12fce-1b31-4a89-a37d-fcf2abb2c2ac' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-a2a12fce-1b31-4a89-a37d-fcf2abb2c2ac' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d89e43da-28b2-450f-b804-c15bdbe63e2e' class='xr-var-data-in' type='checkbox'><label for='data-d89e43da-28b2-450f-b804-c15bdbe63e2e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>travel time over path length</dd><dt><span>units :</span></dt><dd>s/m</dd></dl></div><div class='xr-var-data'><pre>array([[0.00292067, 0.00292142, 0.00292153, ..., 0.00288992, 0.00288847,
        0.00288617],
       [0.00293916, 0.00293866, 0.00293962, ..., 0.00294176, 0.00293989,
        0.00293958],
       [0.00284224, 0.00289408, 0.00293043, ..., 0.00292219, 0.00290983,
        0.00292205],
       ...,
       [0.0029216 , 0.0029223 , 0.00292173, ..., 0.00289784, 0.00289751,
        0.0028974 ],
       [0.00292833, 0.00293443, 0.00293738, ..., 0.00292011, 0.00292029,
        0.00292076],
       [0.00297746, 0.0029774 , 0.0029823 , ..., 0.00294763, 0.00294675,
        0.00294787]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>variableBlock</span></div><div class='xr-var-dims'>(component, frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.002929 0.002931 ... 9.648e-06</div><input id='attrs-e55d59e1-6163-47f6-bd65-c1e9d37c4cf8' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-e55d59e1-6163-47f6-bd65-c1e9d37c4cf8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-56fcd68e-6021-4777-9a97-0a9b3c8fe495' class='xr-var-data-in' type='checkbox'><label for='data-56fcd68e-6021-4777-9a97-0a9b3c8fe495' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Unknown variables from linear system</dd></dl></div><div class='xr-var-data'><pre>array([[ 2.92893583e-03,  2.93078999e-03,  2.93236465e-03,
         2.93113197e-03,  2.93096180e-03,  2.93204473e-03,
         2.93233290e-03,  2.93321194e-03,  2.93305199e-03,
         2.93287090e-03,  2.93248176e-03,  2.93335605e-03,
         2.93329644e-03,  2.93264227e-03,  2.93383276e-03,
         2.93141655e-03,  2.92999655e-03,  2.93087490e-03,
         2.93066744e-03,  2.93200898e-03,  2.93095954e-03,
         2.93038183e-03,  2.93092375e-03,  2.93239604e-03,
         2.93166022e-03,  2.93321778e-03,  2.93196645e-03,
         2.93156360e-03,  2.93195429e-03,  2.92996404e-03,
         2.92978729e-03,  2.93040604e-03,  2.93126530e-03,
         2.92937336e-03,  2.92975017e-03,  2.93069096e-03,
         2.93045908e-03,  2.93115604e-03,  2.93106782e-03,
         2.93131232e-03,  2.93019538e-03,  2.92943312e-03,
         2.93009034e-03,  2.92864984e-03,  2.92922374e-03,
         2.93064554e-03,  2.93048097e-03,  2.92973934e-03,
         2.92989268e-03,  2.92867432e-03,  2.92848322e-03,
         2.92780922e-03,  2.92710995e-03,  2.92748010e-03,
         2.92874804e-03,  2.93019259e-03,  2.93038585e-03,
         2.92881082e-03,  2.92759981e-03,  2.92773493e-03,
...
         1.04677738e-05,  9.17238797e-06,  1.05336435e-05,
         1.22310563e-05,  1.08720151e-05,  1.13145531e-05,
         1.18521507e-05,  1.05594411e-05,  1.11228933e-05,
         1.15119227e-05,  1.12648582e-05,  1.03956824e-05,
         8.56154025e-06,  9.01873011e-06,  9.46190349e-06,
         9.04502362e-06,  7.33303582e-06,  7.62304940e-06,
         6.04636221e-06,  2.77508146e-06,  1.34141245e-06,
         2.14712126e-06,  1.87969109e-06, -1.82838229e-06,
        -3.71155634e-06, -5.07707295e-06, -4.55662174e-07,
        -5.76644580e-08, -1.76034581e-06, -2.34354334e-06,
        -2.64916595e-06,  4.34578430e-07,  4.86868432e-07,
        -3.78634083e-06, -4.11094893e-06, -4.10267958e-06,
        -5.38541292e-06, -3.48140989e-06, -1.64690576e-06,
        -2.96952863e-07, -6.83196346e-07, -2.92704238e-08,
         4.84397323e-07,  2.18768564e-06,  1.95535677e-06,
         1.95228684e-06,  2.61707530e-06,  3.10324463e-06,
         1.26422665e-06,  1.38797531e-06,  1.26193961e-06,
         5.00323747e-06,  5.92807909e-06,  6.55377867e-06,
         8.35235007e-06,  6.94926405e-06,  6.87744282e-06,
         9.83704272e-06,  1.02855114e-05,  9.64800859e-06]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>c</span></div><div class='xr-var-dims'>(frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>341.4 341.2 341.0 ... 341.8 341.5</div><input id='attrs-9bd072f1-b15d-442e-88a2-b5107c940049' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-9bd072f1-b15d-442e-88a2-b5107c940049' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8b10e06f-9c33-4e8a-843d-11775a673d04' class='xr-var-data-in' type='checkbox'><label for='data-8b10e06f-9c33-4e8a-843d-11775a673d04' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Laplace speed of sound from AT array</dd><dt><span>units :</span></dt><dd>m/s</dd></dl></div><div class='xr-var-data'><pre>array([341.42093199, 341.20493201, 341.0217072 , 341.1651236 ,
       341.18493153, 341.05891649, 341.02539998, 340.92319954,
       340.94179156, 340.96284245, 341.00808897, 340.90645139,
       340.91337825, 340.98942414, 340.85105746, 341.13200343,
       341.29733076, 341.19504781, 341.21919986, 341.06307583,
       341.18519469, 341.25245718, 341.18936025, 341.01805653,
       341.10364919, 340.92252107, 341.06802277, 341.1148915 ,
       341.0694374 , 341.30111761, 341.32170707, 341.24963755,
       341.14960477, 341.36993676, 341.32603243, 341.21646147,
       341.24346145, 341.16232216, 341.17258966, 341.14413331,
       341.2741713 , 341.36297368, 341.28640505, 341.45427234,
       341.38737368, 341.22175064, 341.24091247, 341.32729324,
       341.30942931, 341.45141878, 341.47370006, 341.55230924,
       341.63390465, 341.59070801, 341.44282391, 341.27449576,
       341.25198927, 341.43550405, 341.57673993, 341.56097571,
       341.42709767, 341.42110172, 341.41183167, 341.39356256,
       341.54286867, 341.55593177, 341.64325571, 341.51815367,
       341.58174617, 341.65604812, 341.68412126, 341.55634639,
       341.56927095, 341.59494167, 341.46227033, 341.64008511,
       341.65070316, 341.86000649, 341.83502128, 341.70418595,
       341.72768058, 341.66702708, 341.60458979, 341.54119332,
       341.62500099, 341.63268163, 341.66538244, 341.84085355,
       341.63640297, 341.58828041, 341.74424049, 341.66353558,
       341.75758169, 341.91760199, 341.90716854, 341.9533958 ,
       341.94297856, 342.04452848, 341.97205605, 341.94702   ,
       341.95985836, 342.0706732 , 342.00488557, 341.78644419,
       341.73828419, 341.89311502, 341.84702431, 341.75743293,
       341.87516085, 341.96659285, 341.85132702, 341.99380281,
       341.84097895, 341.93258259, 341.82821017, 341.85923016,
       341.98402197, 342.00610805, 341.76439909, 341.48447697])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>u</span></div><div class='xr-var-dims'>(frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.98 4.678 5.111 ... 5.714 6.154</div><input id='attrs-645235d7-ffa2-4781-9d82-974778f6cd6f' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-645235d7-ffa2-4781-9d82-974778f6cd6f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-afd78e09-8f19-4282-9a03-d2233624a0ac' class='xr-var-data-in' type='checkbox'><label for='data-afd78e09-8f19-4282-9a03-d2233624a0ac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>East-West component of velocity from AT array</dd><dt><span>units :</span></dt><dd>m/s</dd></dl></div><div class='xr-var-data'><pre>array([4.97983553, 4.67755781, 5.11099387, 4.79907957, 4.61481005,
       4.42348597, 4.54330998, 5.12350635, 5.10442122, 5.26717979,
       4.86110885, 4.74755254, 4.86075975, 4.60697073, 5.32914961,
       5.10411491, 5.05058487, 5.18404819, 5.11555336, 5.33537011,
       4.83975601, 4.68520658, 4.49577173, 4.89349322, 4.83919568,
       5.62471211, 6.15233708, 5.85749078, 6.01853559, 5.5541219 ,
       5.38048569, 5.02148405, 5.40120409, 5.57459095, 5.71117041,
       5.93806306, 5.92492802, 6.0652497 , 6.29662622, 6.5637584 ,
       6.4315048 , 6.35164445, 6.50108967, 6.5328008 , 6.78835085,
       7.28146164, 7.12212586, 6.92279164, 6.88662284, 6.25537607,
       6.09735895, 6.26643383, 6.74635742, 6.79459744, 6.76910315,
       6.19245513, 5.88488901, 5.71851525, 6.37897517, 6.7417618 ,
       7.39503631, 7.13202014, 6.41340519, 6.361081  , 6.49508752,
       6.55011405, 6.73307903, 6.96977134, 7.06917316, 6.78537006,
       6.75592694, 7.09730907, 7.50381284, 7.42563741, 7.50815794,
       7.5835643 , 7.80321557, 7.73521682, 8.18530845, 8.28521467,
       8.27707148, 8.02140847, 7.85855849, 7.92073125, 8.04077594,
       7.96094655, 7.6839788 , 7.23077131, 7.31946502, 7.45937102,
       7.08522778, 6.78113758, 6.83408671, 7.2638119 , 7.0550971 ,
       7.12789254, 7.20407328, 6.92004851, 6.1490307 , 5.85286522,
       5.81745385, 5.80693566, 5.56253356, 5.47910234, 5.45472942,
       5.78640277, 5.87462593, 5.66816144, 5.99393081, 6.04333439,
       5.99668257, 5.97757194, 6.13421885, 6.2333659 , 6.20832777,
       6.00876933, 5.94994808, 5.81536536, 5.71449173, 6.15405627])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>v</span></div><div class='xr-var-dims'>(frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.815 1.573 1.73 ... 1.201 1.125</div><input id='attrs-fa70d180-b9b9-4169-a6ce-205bc903b59a' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-fa70d180-b9b9-4169-a6ce-205bc903b59a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a4ae8119-de41-4f59-9ed2-0134525f32db' class='xr-var-data-in' type='checkbox'><label for='data-a4ae8119-de41-4f59-9ed2-0134525f32db' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>North-South component of velocity from AT array</dd><dt><span>units :</span></dt><dd>m/s</dd></dl></div><div class='xr-var-data'><pre>array([ 1.81466933,  1.57300719,  1.73004005,  1.69754362,  1.70185762,
        1.58948322,  1.61300839,  1.61054715,  1.71431872,  1.78059644,
        1.8493876 ,  1.72116079,  1.96062084,  1.81677806,  1.8495426 ,
        1.84142836,  1.79579543,  1.88951835,  1.6893876 ,  1.878121  ,
        1.79650417,  1.64161941,  1.60891795,  1.7872932 ,  1.69872393,
        1.59262146,  1.40453263,  1.30743416,  1.27526414,  1.22757702,
        1.20122157,  1.01265812,  1.05014196,  1.0476566 ,  1.05004059,
        1.14014544,  1.30326795,  1.34401524,  1.26132485,  1.23708162,
        1.32325749,  1.47223627,  1.59664398,  1.49131293,  1.46424737,
        1.66225405,  1.48792943,  1.61141248,  1.64994255,  1.7890246 ,
        1.64034638,  1.68148352,  1.36249178,  1.49551953,  1.41134428,
        1.59337648,  1.83035175,  1.49514218,  1.49972226,  1.34253023,
        1.22025417,  1.0692103 ,  1.22782297,  1.42552428,  1.26823721,
        1.31996091,  1.38338438,  1.23159671,  1.29779794,  1.34377356,
        1.3151501 ,  1.21276798,  0.99887119,  1.0523695 ,  1.10322466,
        1.05571659,  0.85595009,  0.89089255,  0.70652457,  0.32402337,
        0.1566472 ,  0.25064711,  0.2193481 , -0.2132815 , -0.43316699,
       -0.59255985, -0.05319183, -0.00673839, -0.20545952, -0.27345062,
       -0.30939378,  0.05073007,  0.05686538, -0.4426522 , -0.48057203,
       -0.47973504, -0.62968941, -0.40730567, -0.19259721, -0.03472203,
       -0.07989062, -0.003425  ,  0.05665867,  0.255561  ,  0.22835645,
        0.22820457,  0.30582982,  0.36245321,  0.14776107,  0.16231143,
        0.1474732 ,  0.58517746,  0.69272719,  0.76625398,  0.97594308,
        0.81214474,  0.80433806,  1.15062096,  1.20137761,  1.12507018])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>T</span></div><div class='xr-var-dims'>(frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>290.1 289.7 289.4 ... 290.6 290.2</div><input id='attrs-74d34990-f775-46c3-9472-74aa7bbd82f9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-74d34990-f775-46c3-9472-74aa7bbd82f9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f5e98bb1-3203-4d54-95ce-9755d3c6392c' class='xr-var-data-in' type='checkbox'><label for='data-f5e98bb1-3203-4d54-95ce-9755d3c6392c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([290.05649631, 289.68960386, 289.37856457, 289.62201158,
       289.65564326, 289.4417169 , 289.38483171, 289.2114087 ,
       289.24295346, 289.27867224, 289.35545316, 289.18299388,
       289.19474579, 289.3237787 , 289.08902276, 289.56578154,
       289.84652177, 289.67282034, 289.71383173, 289.44877665,
       289.6560901 , 289.77030906, 289.66316302, 289.37236894,
       289.51764722, 289.2102576 , 289.45717331, 289.53673175,
       289.45957444, 289.85295376, 289.88792638, 289.76552058,
       289.59566368, 289.96985608, 289.89527357, 289.70918167,
       289.75503205, 289.6172552 , 289.6346879 , 289.5863745 ,
       289.80718679, 289.9580269 , 289.82796475, 290.11314812,
       289.99947971, 289.71816325, 289.75070331, 289.89741523,
       289.8670715 , 290.10829913, 290.14616219, 290.27976413,
       290.41847394, 290.34503678, 290.09369436, 289.80773785,
       289.76951442, 290.08125641, 290.32129212, 290.2944953 ,
       290.06697258, 290.0567847 , 290.04103403, 290.00999438,
       290.26371759, 290.28592163, 290.43437258, 290.22171051,
       290.32980223, 290.4561229 , 290.50385716, 290.28662639,
       290.30859581, 290.35223388, 290.12673909, 290.42898189,
       290.44703502, 290.80301352, 290.76050776, 290.53797664,
       290.57793118, 290.47479054, 290.36863571, 290.26086996,
       290.40333636, 290.4163946 , 290.47199411, 290.77042956,
       290.42272153, 290.34090998, 290.6060943 , 290.46885384,
       290.62878443, 290.90100893, 290.88325579, 290.96191835,
       290.94419094, 291.11702529, 290.99367455, 290.95106834,
       290.97291619, 291.16153098, 291.04954836, 290.67787552,
       290.59596439, 290.85934375, 290.78092737, 290.62853143,
       290.82879618, 290.98437704, 290.78824734, 291.03068559,
       290.77064289, 290.92650027, 290.74892099, 290.80169275,
       291.01403918, 291.05162904, 290.6403795 , 290.1644765 ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ea6adeb4-cb87-4668-b8cb-8c154e86944f' class='xr-section-summary-in' type='checkbox'  ><label for='section-ea6adeb4-cb87-4668-b8cb-8c154e86944f' class='xr-section-summary' >Indexes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>frame</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-0eeedc5d-c481-40cd-a61c-517ed8f06e64' class='xr-index-data-in' type='checkbox'/><label for='index-0eeedc5d-c481-40cd-a61c-517ed8f06e64' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Int64Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
            ...
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
           dtype=&#x27;int64&#x27;, name=&#x27;frame&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>pathID<br>spk<br>mic</div></div><div class='xr-index-preview'>PandasMultiIndex</div><div></div><input id='index-e15c3110-f427-42e6-8147-a54aae9e38ec' class='xr-index-data-in' type='checkbox'/><label for='index-e15c3110-f427-42e6-8147-a54aae9e38ec' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(MultiIndex([(0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 6),
            (5, 7),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6)],
           name=&#x27;pathID&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>variable</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-193ec4ce-2f6f-499d-bc0b-cc02dda81685' class='xr-index-data-in' type='checkbox'/><label for='index-193ec4ce-2f6f-499d-bc0b-cc02dda81685' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;ones&#x27;, &#x27;ncos&#x27;, &#x27;nsin&#x27;], dtype=&#x27;object&#x27;, name=&#x27;variable&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>component</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-25973f68-5834-4f3b-9d5e-b811b5f6bb82' class='xr-index-data-in' type='checkbox'/><label for='index-25973f68-5834-4f3b-9d5e-b811b5f6bb82' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Int64Index([0, 1, 2], dtype=&#x27;int64&#x27;, name=&#x27;component&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-19dda7c1-fcc6-4854-ae32-22a6006bfbbe' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-19dda7c1-fcc6-4854-ae32-22a6006bfbbe' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
ls.ds['time'] = ls.ds['frame']*0.5
ls.ds = ls.ds.swap_dims({'frame':'time'})
```


```python
auxdata.ds['time'] = auxdata.ds.time.values.astype(float)/10**9
```


```python
fig, ax = plt.subplots(3,1, figsize=(5,5), sharex=True)

ls.ds.u.plot(ax=ax[0])
auxdata.ds.u.plot(ax=ax[0])

ls.ds.v.plot(ax=ax[1])
auxdata.ds.v.plot(ax=ax[1])

ls.ds.c.plot(ax=ax[2], label='AT array')
auxdata.ds.c.plot(ax=ax[2], label='Sonic')
ax[2].legend()

fig.tight_layout()
```


    
![png](output_9_0.png)
    

