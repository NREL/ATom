```python
%load_ext autoreload
%autoreload 2

# hardware
import os
# analysis
import xarray as xr
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

# Covariance Matrices
The entire methology of time-dependent stochastic inversion (TDSI) hinges on defining an optimal inverse operator that maps the data vector (containing acoustic signal travel time esimates) onto the model space (where we want to retrieve fluctuating velocity and temperature. 
$$
\mathbf{m} = \mathbf{A}\mathbf{d}
$$
We define the operator $\mathbf{A}$ considering the expected correlation between observed travel times and the background fluctating fields. 

```{note}
Coming innovation in the development of covariance matrices is expected to increase the accuracy of the AT array retrievals and may also provide benefits to the resolved turbulent structures!
```

## Instantiate atarray and constants objects
These objects are used later in the development of the model grid and covariance matrices.
- see `01_signalProcessingModule_20240410.ipynb` for guideance on the atarray object.
- see `03_bulkFieldEstimation_20240412.ipynb` for guideance on the linear system object.




```python
## Array data
atarray = instantiate(cfg.atarray)
atarray.setupPathIntegrals()

## Constants
constants = instantiate(cfg.constants)

## Bulk flow linear system
### instatiate from saved data
ls = atom.backgroundFlow.linearsystem.LinearSystem.from_netcdf('../bulkField_example_data.nc')
```

## Model Grid object
The model grid defines the domain where fluctuating velocity and temperature fields are to be retrieved. Inputs are:
- `nModelPoints(X|Y)`: the integer number of points in the X (easting) and Y (northing) directions.
- `modelLims(X|Y)`: the (min, max) domain limits in the X (easting) and Y (northing) directions. Should be an iterable (list, tuple, array, etc.)
Model grid points are stacked into a dimension denoted `modelXY` with dimension $J = m_x \times m_y$.


```python
# ModelGrid object
mg = atom.fluctuatingField.ModelGrid(
    nModelPointsX=51, 
    nModelPointsY=51, 
    modelLimsX=np.array([-50,50]), 
    modelLimsY=np.array([-50,50])
)
mg.buildModelGrid()
```


```python
mg.ds
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
Dimensions:    (variable: 2, modelXY: 2601, x: 51, y: 51)
Coordinates:
  * variable   (variable) &lt;U1 &#x27;x&#x27; &#x27;y&#x27;
  * x          (modelXY) float64 -50.0 -50.0 -50.0 -50.0 ... 50.0 50.0 50.0 50.0
  * y          (modelXY) float64 -50.0 -48.0 -46.0 -44.0 ... 44.0 46.0 48.0 50.0
  * modelXY    (modelXY) object MultiIndex
Data variables:
    modelGrid  (variable, x, y) float64 -50.0 -48.0 -46.0 ... 50.0 50.0 50.0
Attributes:
    description:  Model grid for TDSI solution
    unit:         m
    nx:           51
    ny:           51</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-2f569508-9f6e-4bdb-b311-3c44387453ab' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-2f569508-9f6e-4bdb-b311-3c44387453ab' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>variable</span>: 2</li><li><span class='xr-has-index'>modelXY</span>: 2601</li><li><span>x</span>: 51</li><li><span>y</span>: 51</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-88954f51-5772-4ea8-a62f-dc5377f71ea5' class='xr-section-summary-in' type='checkbox'  checked><label for='section-88954f51-5772-4ea8-a62f-dc5377f71ea5' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>variable</span></div><div class='xr-var-dims'>(variable)</div><div class='xr-var-dtype'>&lt;U1</div><div class='xr-var-preview xr-preview'>&#x27;x&#x27; &#x27;y&#x27;</div><input id='attrs-c451291e-f1d0-43c3-8caa-f95813c24710' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c451291e-f1d0-43c3-8caa-f95813c24710' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c5fb161c-8f89-4047-8da7-4a4ae7df2cea' class='xr-var-data-in' type='checkbox'><label for='data-c5fb161c-8f89-4047-8da7-4a4ae7df2cea' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;x&#x27;, &#x27;y&#x27;], dtype=&#x27;&lt;U1&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(modelXY)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-50.0 -50.0 -50.0 ... 50.0 50.0</div><input id='attrs-b9fa578a-d40d-402d-9bfc-46a5e1e33a87' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b9fa578a-d40d-402d-9bfc-46a5e1e33a87' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1bf607f9-5c26-41ea-be5f-14a21e57f15c' class='xr-var-data-in' type='checkbox'><label for='data-1bf607f9-5c26-41ea-be5f-14a21e57f15c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-50., -50., -50., ...,  50.,  50.,  50.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(modelXY)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-50.0 -48.0 -46.0 ... 48.0 50.0</div><input id='attrs-f14e4c3b-55db-4b76-b4d2-385c81d3f4c8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f14e4c3b-55db-4b76-b4d2-385c81d3f4c8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-74d93d2f-a7e0-47b8-888a-05592fee348c' class='xr-var-data-in' type='checkbox'><label for='data-74d93d2f-a7e0-47b8-888a-05592fee348c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-50., -48., -46., ...,  46.,  48.,  50.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>modelXY</span></div><div class='xr-var-dims'>(modelXY)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>MultiIndex</div><input id='attrs-21e36dff-0b7e-47de-b6c8-a6608cb59135' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-21e36dff-0b7e-47de-b6c8-a6608cb59135' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2d6c2cb7-160e-4a4b-ba60-da5bbd93df1d' class='xr-var-data-in' type='checkbox'><label for='data-2d6c2cb7-160e-4a4b-ba60-da5bbd93df1d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([(-50.0, -50.0), (-50.0, -48.0), (-50.0, -46.0), ..., (50.0, 46.0),
       (50.0, 48.0), (50.0, 50.0)], dtype=object)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4dffb525-2d31-482e-8648-c4f4c6f17622' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4dffb525-2d31-482e-8648-c4f4c6f17622' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>modelGrid</span></div><div class='xr-var-dims'>(variable, x, y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-50.0 -48.0 -46.0 ... 50.0 50.0</div><input id='attrs-df39b1d4-8030-4e42-bb84-cbb570d7e5b4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-df39b1d4-8030-4e42-bb84-cbb570d7e5b4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5820ce4f-13c4-4cb7-9440-2b7fff495f5c' class='xr-var-data-in' type='checkbox'><label for='data-5820ce4f-13c4-4cb7-9440-2b7fff495f5c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-50., -48., -46., ...,  46.,  48.,  50.],
        [-50., -48., -46., ...,  46.,  48.,  50.],
        [-50., -48., -46., ...,  46.,  48.,  50.],
        ...,
        [-50., -48., -46., ...,  46.,  48.,  50.],
        [-50., -48., -46., ...,  46.,  48.,  50.],
        [-50., -48., -46., ...,  46.,  48.,  50.]],

       [[-50., -50., -50., ..., -50., -50., -50.],
        [-48., -48., -48., ..., -48., -48., -48.],
        [-46., -46., -46., ..., -46., -46., -46.],
        ...,
        [ 46.,  46.,  46., ...,  46.,  46.,  46.],
        [ 48.,  48.,  48., ...,  48.,  48.,  48.],
        [ 50.,  50.,  50., ...,  50.,  50.,  50.]]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e22bdaf8-be37-42a2-bf04-35bc96ef4110' class='xr-section-summary-in' type='checkbox'  ><label for='section-e22bdaf8-be37-42a2-bf04-35bc96ef4110' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>variable</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-776a103b-d691-48de-9346-5e93b9f77229' class='xr-index-data-in' type='checkbox'/><label for='index-776a103b-d691-48de-9346-5e93b9f77229' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;x&#x27;, &#x27;y&#x27;], dtype=&#x27;object&#x27;, name=&#x27;variable&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>x<br>y<br>modelXY</div></div><div class='xr-index-preview'>PandasMultiIndex</div><div></div><input id='index-f9f72abc-588a-4ac4-8a85-ba0ce2f41bdb' class='xr-index-data-in' type='checkbox'/><label for='index-f9f72abc-588a-4ac4-8a85-ba0ce2f41bdb' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(MultiIndex([(-50.0, -50.0),
            (-50.0, -48.0),
            (-50.0, -46.0),
            (-50.0, -44.0),
            (-50.0, -42.0),
            (-50.0, -40.0),
            (-50.0, -38.0),
            (-50.0, -36.0),
            (-50.0, -34.0),
            (-50.0, -32.0),
            ...
            ( 50.0,  32.0),
            ( 50.0,  34.0),
            ( 50.0,  36.0),
            ( 50.0,  38.0),
            ( 50.0,  40.0),
            ( 50.0,  42.0),
            ( 50.0,  44.0),
            ( 50.0,  46.0),
            ( 50.0,  48.0),
            ( 50.0,  50.0)],
           name=&#x27;modelXY&#x27;, length=2601))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-83bbc105-189e-4413-8e00-1dd3da208afa' class='xr-section-summary-in' type='checkbox'  checked><label for='section-83bbc105-189e-4413-8e00-1dd3da208afa' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Model grid for TDSI solution</dd><dt><span>unit :</span></dt><dd>m</dd><dt><span>nx :</span></dt><dd>51</dd><dt><span>ny :</span></dt><dd>51</dd></dl></div></li></ul></div></div>



## Covariance Matrix object
Configuration data comes from the hydra configuration object.

Default values:
- `sigmaT`: 0.14, standard devaition of the temperature fluctuations
- `lT`: 15.0, length scale of the temperature fluctuations
- `sigmaU`: 0.72, standard devaition of the u velocity fluctuations
- `sigmaV`: 0.42, standard devaition of the v velocity fluctuations
- `l`: 15.0, length scale of the velocity fluctuations
- `timeDelay`: 0.0, time delay between successive frames. Will be updated to generalize to variable time delays between frames
- `nFrames`: 0, the number of additional frames considered in the covariance matrices. A value of 0 implies simple stochastic inversion without time dependence.
- `alignment`: "center", alignment of frames in the covariance matrices (also accepts "forward" or "backward")
- `advectionScheme`: "stationary", simplifying assumption of statistical stationarity for advection (constant advection velocity and time delay). Will be updated to include a "direct" method where the advection velocity varies between frames.

This example updates nFrames to 2 to use one frame before and one frame after the target frame in the fluctuating field retrieval.


```python
# cfg.covariancematrix['nFrames'] = 2

cm = atom.fluctuatingField.CovarianceMatrices(
    cfg.covariancematrix,
    mg.ds, 
    atarray.ds,
    ls.ds,
)
cm.assembleTDSICovarianceMatrices()
# cm.to_pickle(f'../covarianceMatrices_nF={cfg.covariancematrix.nFrames}_example.pk')

# cm = atom.fluctuatingField.CovarianceMatrices.from_pickle('covarianceMatrices_nF=6.pk')
```

The model-data and data-data covariance matrices $B_{md}$ and $B_{dd}$ rely on an assumed (or imposed) distribution of covariances. Covariance functions relate fluctuations between any two points $\mathbf{r}$  and $\mathbf{r^\prime}$ with a Gaussian distribution.

$\begin{align}
    B^S_{TT} &= \sigma_T^2 \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l_T^2}\right)\label{eq:corr1},\\
    B^S_{uu} &= \sigma_u^2 \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l^2}\right) \left( 1 - \frac{(y - y^\prime)^2}{l^2} \right)\label{eq:corr2},\\
    B^S_{vv} &= \sigma_v^2 \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l^2}\right)\left( 1 - \frac{(x - x^\prime)^2}{l^2} \right)\label{eq:corr3},\\
    B^S_{uv} &= \sigma_u \sigma_v \text{exp}\left( -\frac{(\mathbf{r} - \mathbf{r}^\prime)^2}{l^2}\right) \left( \frac{(x - x^\prime)(y - y^\prime)}{l^2} \right)\label{eq:corr4},
\end{align}$

Other distributions or covariance models can be added by replicating the structure found in `CovarianceMatrix.covarianceFunction` method. For heterogeneous fields, this function can be replaced by input data, so that $B_{md}$ and $B_{dd}$ map observed covariances to observed travel times directly.

When considering multiple frames, as in the TDSI, the second point $\mathbf{R}^\prime$ takes into account advection by the mean velocity over the time delta between frames. The covariance matrix calculation is centered by default, meaning advection takes place in a forward and a backward sense collecting the frames $[\tau - N/2 * \Delta t,...,\tau,...,\tau + N/2* \Delta t]$. The algorithm takes into account edge cases at the beginning and end of the set of frames. These are defined in the `stencil` and `frameSets` attributes.


```python
fig, ax = plt.subplots(2,2, figsize=(8,6))
ax=ax.flatten()

pointID = np.random.randint(len(cm.ds.modelXY))

cm.ds.B_uu.isel(modelXY=pointID).T.plot(ax=ax[0])
cm.ds.B_vv.isel(modelXY=pointID).T.plot(ax=ax[1])
cm.ds.B_uv.isel(modelXY=pointID).T.plot(ax=ax[2])
cm.ds.B_TT.isel(modelXY=pointID).T.plot(ax=ax[3])

fig.tight_layout()
```


    
![png](output_9_0.png)
    


### Model-data covariance matrices

The model data covariance matrix describes correlation between observed travel times and the modeled turbulence fields. $\mathbf{B_{md}}$ has a size of $[ N\times I, 3\times J]$ considering $N$ frames in the TDSI, $I$ acoustic travel paths, and $J$ modeled points for each of the three variables ($u$, $v$ and $T$).

$\begin{eqnarray}    
    \mathbf{B}_{\mathbf{m}_j\mathbf{d}_{0i}}(t_1, t_2) &=& \langle m_j(t_1) d_{0i}(t_2) \rangle\\
    &=& \int_{L_i} \left( \frac{c_0(t_2)}{2T_0(t_2)} \langle m_j(t_1) T(\mathbf{r},t_2) \rangle + \langle m_j(t_1) u(\mathbf{r},t_2)\rangle \text{cos}~\phi_i + \langle m_j(t_1) v(\mathbf{r},t_2)\rangle \text{sin}~\phi_i \right) dl\\
    &=& \begin{cases}
        \int_{L_i} \left( \frac{c_0(t_2)}{2T_0(t_2)}B_{TT}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \right) dl, & \text{if } 1 \le j \le J \\
        \int_{L_i} \left(B_{uu}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{cos}~\phi_i + B_{uv}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{sin}~\phi_i \right) dl, & \text{if } J+1 \le j \le 2J \\
        \int_{L_i} \left(B_{vu}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{cos}~\phi_i + B_{vv}(\mathbf{r}_j, t_1; \mathbf{r}, t_2) \text{sin}~\phi_i \right) dl, & \text{if } 2J+1 \le j \le 3J
  \end{cases}
\end{eqnarray}$


```python
cm.ds['modelXY'] = mg.ds.modelXY

fig, ax = plt.subplots(1,3, figsize=(12,3.5))
cm.ds.Bmd.sel(component='u', tdsiFrame=0).sum(dim='pathID').unstack().plot(ax=ax[0])
cm.ds.Bmd.sel(component='v', tdsiFrame=0).sum(dim='pathID').unstack().plot(ax=ax[1])
cm.ds.Bmd.sel(component='T', tdsiFrame=0).sum(dim='pathID').unstack().plot(ax=ax[2])

fig.tight_layout()
```


    
![png](output_11_0.png)
    


### Data-data covariance matrix

The data-data covariance matrix describes auto correlation between observed data. $\mathbf{R_{dd}}$ has a size of $[ N\times I,  N\times I]$ considering $N$ frames in the TDSI and $I$ acoustic travel paths.

$B_{dd} = \langle d_i(t_1) d_p(t_2) \rangle = \iint_{L_i, L_p}\frac{c_0(t_1) c_0(t_2)}{4 T_0(t_1)T_0(t_2)}B_{TT} + B_{uu} \cos\varphi_i\cos\varphi_p + B_{vv} \sin\varphi_i\sin\varphi_p + B_{uv} \cos\varphi_i\sin\varphi_p + B_{vu} \sin\varphi_i\cos\varphi_p d l_i d l_p$



```python
plt.pcolor(cm.ds.Bdd.unstack().stack(NI = ['tdsiFrame', 'spk','mic'],
                          NI_duplicate = ['tdsiFrame_duplicate', 'spk_duplicate', 'mic_duplicate']).values)
              
```




    <matplotlib.collections.PolyCollection at 0x7fea32304310>




    
![png](output_13_1.png)
    

