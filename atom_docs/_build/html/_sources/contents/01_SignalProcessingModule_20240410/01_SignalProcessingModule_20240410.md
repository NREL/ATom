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

## The `atom` Signal Processing Module
This is the entry point to the code base for field data. 
- the array facility is described in the configuration data (number of measurement points, point locations, etc.)
- data recorded at the AT array facility at the NREL Flatirons Campus include time series data from a sonic anemometer, and records of of the emitted chirps and microphone recordings

## Acoustic tomography array object
- instatiate from configuration object
- calculates path info (length, orientation, unit vector) internally
- path integral information setup for later use
- plot speaker and microphone locations
- plot acoustic travel paths


```python
### Array data
atarray = instantiate(cfg.atarray)
atarray.setupPathIntegrals()
```


```python
fig, ax = plt.subplots(1,2, figsize=(7,3.5), sharex=True, sharey=True)

ax[0] = atarray.plotSpkMicLocations(ax=ax[0])

c = plt.cm.viridis(np.linspace(0, 1, len(atarray.ds.pathID)))
ax[1] = atarray.plotPaths(ax=ax[1], c=c)

fig.tight_layout()
```


    
![png](output_3_0.png)
    


## Auxiliary data object
- instatiate from configuration object
- plot time series
- calculate and plot spectra with Welch's method
- calculate and plot autocorrelation coefficient
- estimate integral time and length scales (Taylor's hypothesis)


```python
### Auxiliary data
auxdata = instantiate(cfg.auxdata)
auxDataPath = "../AuxData_example.txt"
auxdata.loadData(auxDataPath)


fig, ax = plt.subplots(5,1, figsize=(5, 6), sharex=True)

for ii, var in enumerate(['u','v','c','T','H']):
    auxdata.ds[var].plot(ax=ax[ii], c=f'C{ii}')
fig.tight_layout()
```


    
![png](output_5_0.png)
    



```python
auxdata.calculateSpectra()

fig, ax = plt.subplots(5,1, figsize=(5, 6), sharex=True)

for ii, var in enumerate(['u','v','uz','T','H']):
    ax[ii].loglog(auxdata.ds.frequency, auxdata.ds[f'S_{var}'], c=f'C{ii}')
    ax[ii].set_ylabel(f'$S_{{{var}}}$ [{auxdata.ds[f"S_{var}"].units}]')
ax[-1].set_xlabel('Frequency [Hz]')
fig.tight_layout()

```


    
![png](output_6_0.png)
    



```python
auxdata.autoCorrelation()

fig, ax = plt.subplots(figsize=(5, 3.5))

for ii, var in enumerate(['u','v','uz','T','H']):
    ax.plot(auxdata.ds.time, auxdata.ds[f'rho_{var}'], c=f'C{ii}', label=f'$\\rho_{{{var}}}$')
ax.set_ylabel(f'Autocorrelation Coefficient [-]')
ax.set_xlabel('Time lag [s]')

ax.legend()

fig.tight_layout()

```


    
![png](output_7_0.png)
    



```python
auxdata.integral_scale()
auxdata.ds.attrs
```




    {'samplingFrequency': 20.0,
     'recordTimeDuration': 0.5,
     'recordTimeDelta': 0.05,
     'recordLength': 1200,
     'sonicAnemometerOrientation': 0.0,
     'windowType': 'hann',
     'threshold': 0.5,
     'tau_u': 7.918675695892259,
     'L_u': 48.17162047075365,
     'tau_v': 7.296752778678485,
     'L_v': 13.957173989410366,
     'tau_uz': 4.958264552346053,
     'L_uz': 0.43593888321651886,
     'tau_T': 1.1152540937344828,
     'L_T': 33.19953070966245,
     'tau_H': 12.001672723618048,
     'L_H': 152.64853526854668}



## Audio data object
- instatiate from configuration object
- load data
- isolate reference signal emitted by each speaker
- plot recorded microphone data


```python
### Microphone data
audiodata = instantiate(cfg.audiodata)
mainDataPath = "../MainData_example.txt"
audiodata.loadData(mainDataPath)
```


```python
audiodata.ds
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
Dimensions:                    (time: 10000, mic: 8, frame: 120, spk: 8)
Coordinates:
  * time                       (time) timedelta64[ns] 00:00:00 ... 00:00:00.4...
  * mic                        (mic) int64 0 1 2 3 4 5 6 7
  * frame                      (frame) int64 0 1 2 3 4 5 ... 115 116 117 118 119
  * spk                        (spk) int64 0 1 2 3 4 5 6 7
Data variables:
    speakerSignalEmissionTime  (spk) float64 0.124 0.104 0.204 ... 0.04 0.144
    micData                    (time, mic, frame) float64 -5.884 ... -9.248
    refSig                     (time) float64 0.0 0.0 0.0 0.0 ... nan nan nan
Attributes:
    description:            Signals recorded by microphones
    samplingFrequency:      20000
    deltaT:                 0.05
    recordTimeDuration:     0.5
    recordTimeDelta:        5e-05
    recordLength:           10000
    nMics:                  8
    nFrames:                120
    chirpCentralFrequency:  1200
    chirpBandwidth:         700
    nSpeakers:              8</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-0209ab37-0ca4-416f-ae28-a882e089fb84' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-0209ab37-0ca4-416f-ae28-a882e089fb84' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 10000</li><li><span class='xr-has-index'>mic</span>: 8</li><li><span class='xr-has-index'>frame</span>: 120</li><li><span class='xr-has-index'>spk</span>: 8</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6ce1adcb-0c63-4f7a-9eb7-eae5d75c5ab9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6ce1adcb-0c63-4f7a-9eb7-eae5d75c5ab9' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>timedelta64[ns]</div><div class='xr-var-preview xr-preview'>00:00:00 ... 00:00:00.499950</div><input id='attrs-6bbf9d28-25dc-4d85-8d64-3d1a67d59147' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6bbf9d28-25dc-4d85-8d64-3d1a67d59147' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0c49ba01-ddbd-46bb-97bf-caefedd554c4' class='xr-var-data-in' type='checkbox'><label for='data-0c49ba01-ddbd-46bb-97bf-caefedd554c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([        0,     50000,    100000, ..., 499850000, 499900000, 499950000],
      dtype=&#x27;timedelta64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>mic</span></div><div class='xr-var-dims'>(mic)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-5da40e61-6c35-4854-9858-50221494f41b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5da40e61-6c35-4854-9858-50221494f41b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-684efc99-1ed6-48ae-affa-810a5c86567d' class='xr-var-data-in' type='checkbox'><label for='data-684efc99-1ed6-48ae-affa-810a5c86567d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>frame</span></div><div class='xr-var-dims'>(frame)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-cd74d12e-1f35-40c5-b031-5def44163465' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cd74d12e-1f35-40c5-b031-5def44163465' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-47b54e20-9cec-404f-b048-4f70b7bb2615' class='xr-var-data-in' type='checkbox'><label for='data-47b54e20-9cec-404f-b048-4f70b7bb2615' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>spk</span></div><div class='xr-var-dims'>(spk)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-73873268-744f-432e-803e-ad9b0579eace' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-73873268-744f-432e-803e-ad9b0579eace' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f43ae42c-133b-4934-aa33-c15b5d201f16' class='xr-var-data-in' type='checkbox'><label for='data-f43ae42c-133b-4934-aa33-c15b5d201f16' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7ddabb48-6e39-4e1b-94e2-adcc7b4e56e5' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7ddabb48-6e39-4e1b-94e2-adcc7b4e56e5' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>speakerSignalEmissionTime</span></div><div class='xr-var-dims'>(spk)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.124 0.104 0.204 ... 0.04 0.144</div><input id='attrs-6d80f626-41fe-4c93-b25c-9872b0b7c449' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-6d80f626-41fe-4c93-b25c-9872b0b7c449' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bea6f5bc-301e-477f-b371-910770e75fbb' class='xr-var-data-in' type='checkbox'><label for='data-bea6f5bc-301e-477f-b371-910770e75fbb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Speaker signal emission time</dd><dt><span>units :</span></dt><dd>seconds</dd></dl></div><div class='xr-var-data'><pre>array([0.124, 0.104, 0.204, 0.   , 0.16 , 0.2  , 0.04 , 0.144])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>micData</span></div><div class='xr-var-dims'>(time, mic, frame)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-5.884 -0.527 ... -6.017 -9.248</div><input id='attrs-000a8dbc-794c-4b65-821e-c0787e548c36' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-000a8dbc-794c-4b65-821e-c0787e548c36' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4ca7ded0-ca1a-4c26-abd4-66fc66ed4adf' class='xr-var-data-in' type='checkbox'><label for='data-4ca7ded0-ca1a-4c26-abd4-66fc66ed4adf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-5.884, -0.527, -4.704, ...,  2.797, -3.62 ,  0.11 ],
        [-7.113,  3.103, -3.107, ..., -2.802, -6.434, -2.686],
        [-0.316,  1.218,  8.626, ..., -0.21 ,  1.147,  7.356],
        ...,
        [-0.475, -5.393, -0.173, ..., 10.237,  0.133, -5.965],
        [-1.276, -0.084, -4.247, ...,  0.164,  7.261, -4.166],
        [-5.958,  2.175,  4.393, ..., -4.206, -4.921, -5.649]],

       [[-5.939, -0.529, -4.719, ...,  2.783, -3.611,  0.101],
        [-7.11 ,  3.158, -3.13 , ..., -2.804, -6.441, -2.654],
        [-0.302,  1.246,  8.63 , ..., -0.198,  1.121,  7.36 ],
        ...,
        [-0.484, -5.419, -0.155, ..., 10.182,  0.121, -5.987],
        [-1.26 , -0.082, -4.245, ...,  0.194,  7.225, -4.2  ],
        [-5.96 ,  2.166,  4.33 , ..., -4.206, -4.918, -5.654]],

       [[-5.983, -0.513, -4.749, ...,  2.799, -3.602,  0.108],
        [-7.111,  3.156, -3.112, ..., -2.781, -6.449, -2.683],
        [-0.277,  1.262,  8.633, ..., -0.18 ,  1.105,  7.381],
        ...,
...
        ...,
        [-4.93 , -0.521, -1.56 , ..., -2.105,  2.458,  0.35 ],
        [-0.233, -3.705, -1.202, ...,  7.977, -6.414, -6.014],
        [ 0.423, 10.236,  5.78 , ..., -3.693, -5.952, -9.263]],

       [[-0.432, -2.493,  2.035, ..., -1.887,  0.733,  3.316],
        [ 1.74 , -2.615, -0.131, ..., -5.212, -2.899, -4.318],
        [ 1.678,  8.965, -2.219, ...,  3.36 ,  6.771,  1.492],
        ...,
        [-4.934, -0.533, -1.576, ..., -2.137,  2.43 ,  0.365],
        [-0.212, -3.694, -1.174, ...,  7.975, -6.421, -6.032],
        [ 0.42 , 10.236,  5.767, ..., -3.672, -5.989, -9.248]],

       [[-0.425, -2.482,  2.04 , ..., -1.891,  0.734,  3.352],
        [ 1.742, -2.641, -0.148, ..., -5.205, -2.915, -4.339],
        [ 1.688,  8.955, -2.201, ...,  3.372,  6.8  ,  1.494],
        ...,
        [-4.916, -0.555, -1.589, ..., -2.163,  2.392,  0.415],
        [-0.225, -3.71 , -1.168, ...,  7.97 , -6.356, -6.036],
        [ 0.435, 10.236,  5.726, ..., -3.701, -6.017, -9.248]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>refSig</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.0 0.0 0.0 ... nan nan nan nan</div><input id='attrs-1398a9ef-586f-49f9-b367-9b66b017435d' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-1398a9ef-586f-49f9-b367-9b66b017435d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-36aaecb0-ecd0-4ac1-8ce7-e91a0ccae3c8' class='xr-var-data-in' type='checkbox'><label for='data-36aaecb0-ecd0-4ac1-8ce7-e91a0ccae3c8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Reference chirp signal</dd><dt><span>chirpTimeDuration :</span></dt><dd>0.0058</dd><dt><span>chirpRecordLength :</span></dt><dd>116</dd><dt><span>chirpCentralFrequency :</span></dt><dd>1200</dd><dt><span>chirpBandwidth :</span></dt><dd>700</dd><dt><span>windowHalfWidth :</span></dt><dd>0.01</dd></dl></div><div class='xr-var-data'><pre>array([ 0.,  0.,  0., ..., nan, nan, nan])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2b1d5f65-bc97-422d-afef-6a16ed061b26' class='xr-section-summary-in' type='checkbox'  ><label for='section-2b1d5f65-bc97-422d-afef-6a16ed061b26' class='xr-section-summary' >Indexes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-dff4dcc4-84c0-4356-aba5-a0611ab3696e' class='xr-index-data-in' type='checkbox'/><label for='index-dff4dcc4-84c0-4356-aba5-a0611ab3696e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(TimedeltaIndex([       &#x27;0 days 00:00:00&#x27;, &#x27;0 days 00:00:00.000050&#x27;,
                &#x27;0 days 00:00:00.000100&#x27;, &#x27;0 days 00:00:00.000150&#x27;,
                &#x27;0 days 00:00:00.000200&#x27;, &#x27;0 days 00:00:00.000250&#x27;,
                &#x27;0 days 00:00:00.000300&#x27;, &#x27;0 days 00:00:00.000350&#x27;,
                &#x27;0 days 00:00:00.000400&#x27;, &#x27;0 days 00:00:00.000450&#x27;,
                ...
                &#x27;0 days 00:00:00.499500&#x27;, &#x27;0 days 00:00:00.499550&#x27;,
                &#x27;0 days 00:00:00.499600&#x27;, &#x27;0 days 00:00:00.499650&#x27;,
                &#x27;0 days 00:00:00.499700&#x27;, &#x27;0 days 00:00:00.499750&#x27;,
                &#x27;0 days 00:00:00.499800&#x27;, &#x27;0 days 00:00:00.499850&#x27;,
                &#x27;0 days 00:00:00.499900&#x27;, &#x27;0 days 00:00:00.499950&#x27;],
               dtype=&#x27;timedelta64[ns]&#x27;, name=&#x27;time&#x27;, length=10000, freq=None))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>mic</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-97882600-205c-44c0-a5a3-510d0907c477' class='xr-index-data-in' type='checkbox'/><label for='index-97882600-205c-44c0-a5a3-510d0907c477' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Int64Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;mic&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>frame</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-3f9e09ef-8705-4d0f-bacd-e940b2004993' class='xr-index-data-in' type='checkbox'/><label for='index-3f9e09ef-8705-4d0f-bacd-e940b2004993' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Int64Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
            ...
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
           dtype=&#x27;int64&#x27;, name=&#x27;frame&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>spk</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-307e66bd-6c38-49a0-82d5-6302d3ea9000' class='xr-index-data-in' type='checkbox'/><label for='index-307e66bd-6c38-49a0-82d5-6302d3ea9000' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Int64Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;spk&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9166bb24-6ff8-42ba-8d69-93710c1dafd1' class='xr-section-summary-in' type='checkbox'  ><label for='section-9166bb24-6ff8-42ba-8d69-93710c1dafd1' class='xr-section-summary' >Attributes: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Signals recorded by microphones</dd><dt><span>samplingFrequency :</span></dt><dd>20000</dd><dt><span>deltaT :</span></dt><dd>0.05</dd><dt><span>recordTimeDuration :</span></dt><dd>0.5</dd><dt><span>recordTimeDelta :</span></dt><dd>5e-05</dd><dt><span>recordLength :</span></dt><dd>10000</dd><dt><span>nMics :</span></dt><dd>8</dd><dt><span>nFrames :</span></dt><dd>120</dd><dt><span>chirpCentralFrequency :</span></dt><dd>1200</dd><dt><span>chirpBandwidth :</span></dt><dd>700</dd><dt><span>nSpeakers :</span></dt><dd>8</dd></dl></div></li></ul></div></div>




```python
fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(audiodata.ds.time, audiodata.ds.refSig, label='Reference Signal')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Intensity [-]')
```




    Text(0, 0.5, 'Intensity [-]')




    
![png](output_12_1.png)
    



```python
fig,ax = plt.subplots(figsize=(6,3), sharex=True, sharey=True)

c=plt.cm.viridis(np.linspace(0,1,audiodata.ds.nMics))

for ii, mic in enumerate(audiodata.ds.mic):
    audiodata.ds.micData.sel(frame=0, mic=mic).plot(ax=ax, label=f'Mic {ii}', c=c[ii,:])
ax.set_title('Microphone data')
ax.set_ylabel(f'Intensity [dB]')
ax.legend(loc=6, bbox_to_anchor=(1,0.5))
fig.tight_layout()
```


    
![png](output_13_0.png)
    



```python
fig,ax = plt.subplots(figsize=(6,3), sharex=True, sharey=True)

c=plt.cm.viridis(np.linspace(0,1,audiodata.ds.nMics))

micNumber = 4
audiodata.ds.micData.sel(frame=0, mic=micNumber).plot(ax=ax, label=f'Mic {micNumber}', c=c[micNumber,:])
ax.set_title('Microphone data')
ax.set_ylabel(f'Intensity [dB]')
ax.legend(loc=6, bbox_to_anchor=(1,0.5))
fig.tight_layout()
```


    
![png](output_14_0.png)
    

