import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

import july
import warnings
warnings.filterwarnings('ignore')

################################## Color palette ##########################

colors = ['#084594',"#7fcdbb", '#636363','#edf8b1',"#4374B3"]
custom_palette=sns.color_palette(colors)
customPalette = sns.set_palette(sns.color_palette(colors))

def show_palette():
 sns.set(style="white", palette=customPalette)
 sns.palplot(custom_palette)
 return plt.show()

def change_color_palette(list):
 global  colors,custom_palette, customPalette
 colors = list
 custom_palette=sns.color_palette(colors)
 customPalette = sns.set_palette(sns.color_palette(colors))
 sns.set(style="white", palette=customPalette)
 sns.palplot(custom_palette)
 return plt.show()

#################### Distribution graphics ###############################

def plot_sns_distribution_hue(df_tmp, var_row, var_hue, x_var, title_text, bw=0.2, aspect = 4, height = 1.5, show_label = True):
  '''
  plot_distribution_hue: permite visualizar varias distribuciones simultaneamente, 
  separadas por subset de datos. Ejemplo: Niveles de oxígeno por momento del día. 

  parametros
  - df_tmp: dataframe con set de datos
  - var_row: string, nombre de la variable en el dataframe por la cual se separara el conjunto de datos, debe ser una variable de tipo categoria. Ej: momento del día(mañana, tarde, noche), tipo de ciclo(normal, continuo), etc...
  - var_hue: puede ser la misma variable de var_row o separacion por un segundo criterio.
  - x_var: string, nombre de la variable en estudio
  - title_text: titulo de la figura
  - bw: bandwidth
  - aspect: define el ancho de la figura
  - height: define la altura de cada subplot
  - show_label: Boolean, utilizar cuando var_row es igual a var_hue, muestra el nombre de la categoria
  '''
  g = sns.FacetGrid(df_tmp, #the dataframe to pull from
                    row= var_row, #define the column for each subplot row to be differentiated by
                    hue= var_hue, #define the column for each subplot color to be differentiated by
                    aspect=aspect, #aspect * height = width
                    height=height, #height of each subplot
                    
                  )
  g.fig.suptitle(title_text)
  
  g.map(sns.kdeplot, x_var, shade=True, alpha=0.8, lw=0.5, bw=bw)
  def label( x, color, label):
      ax = plt.gca() #get the axes of the current object
      ax.text(0, .2, #location of text
                label, #text label
                fontweight="bold", color=color, size=15, #text attributes
                ha="left", va="center", #alignment specifications
                transform=ax.transAxes) #specify axes of transformation
  if show_label == True:
        g.map(label, x_var) #the function counts as a plotting object!
  #prevent overlapping issues by 'removing' axis face color
  #sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
  g.fig.subplots_adjust(hspace= 0)
  g.set_titles("") #set title to blank
  g.set_ylabels("")
  g.set(yticks=[]) #set y ticks to blank
  g.despine(bottom=False, left=True) #remove 'spines'
  return plt.show()


def simple_sns_kdeplot(df_dist, column_name):
  '''''
   simple_kdeplot: permite realizar analisis univariado de la distribucion de observaciones en un conjunto ded datos, 
                la representacion gráfica se realiza por medio de la curva de densidad de probabilidad contínua.
                Adicionalmente entrega estadisticos como la media y valores que se encuentran por debajo y arriba de
                una desviacion estandar
        
    Parametros:
    df_dist: dataframe donde está la data
    column_name: columna   
  '''''
  plt.figure(figsize=(6,3))
  ax=sns.kdeplot(df_dist[column_name], shade=False)
  kdeline = ax.lines[0]
  xs = kdeline.get_xdata()
  ys = kdeline.get_ydata()
    
  middle = round(df_dist[column_name].mean(),2)
  sdev = df_dist[column_name].std()
  left = round(middle - sdev,2)
  right = round(middle + sdev,2)
          
  ax.vlines(middle, 0, np.interp(middle, xs, ys),  ls='--', label = f'mean {middle}')
  ax.vlines(left, 0, np.interp(left, xs, ys), ls=':', label = f'left {left}')
  ax.vlines(right, 0, np.interp(right, xs, ys),  ls=':', label = f'right {right}')
  ax.fill_between(xs, 0, ys,  alpha=0.2)
  ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=True, alpha=0.2) 
  ax.legend() 
 
  return plt.show()

def two_sns_histplot(data,variable_name, largo, alto):
  '''''
   two_histplot: Grafica dos histogramas de frecuencia de la variable_name. El primer histograma con el porcentaje de distribución,
    el segundo con la frecuencia de las observaciones.              
        
  Parametros:
    data: dataframe donde está la data
    variable_name: string, nombre de la columna  
    largo: largo de la figura
    alto: alto de la figura 
  '''''

  fig, ax = plt.subplots(1,2,figsize=(largo, alto))
  sns.histplot(data=data, x=variable_name, stat="count", ax=ax[0], bins=50)
  sns.histplot(data=data, x=variable_name, stat="probability", bins=50, ax=ax[1])
  ax[0].set_title('Distribución por cantidades de ' + variable_name )
  ax[1].set_title('Porcentaje distribución de '+ variable_name)
  return plt.show()

def simple_histplot(data,variable_name, largo, alto):
  '''''
   histplot: Grafica un histogramas de frecuencia de la variable_name.
  Parametros:
    data: dataframe donde está la data
    variable_name: columna  
    largo: largo de la figura
    alto: alto de la figura  
  '''''
  fig = plt.figure(figsize=(largo, alto))
  data[variable_name].hist(bins=50)
  plt.title('Distribución  de ' + variable_name )
 
  return plt.show()
#################### lineplots  ###############################

def plot_two_series(df, largo, alto):
  '''''
    Devuelve una gráfica con dos series temporales.   
        
    Parametros:
    df: dataframe con las dos series, cuyo index son valores temporales Ejemplo: date_time
    alto: alto de la figura 
    largo: largo de la figura
 '''''
  fig = plt.figure(figsize=(20,4))
  ax = fig.add_subplot(111)
  ax.plot(df.index,df.iloc[:, 0], color=colors[0], label = df.columns[0])

  ax2 = ax.twinx()
  ax2.plot(df.index,df.iloc[:, 1], color=colors[1], label = df.columns[1])
  fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

  ax.set_ylabel(r" "+df.columns[0])
  ax2.set_ylabel(r" "+ df.columns[1])

  return plt.show()

def multiple_lineplot_secundary_y_axis(df, var_y,list,alto,largo):
 '''''
    Devuelve i-gráficos según la cantidad de elementos en list. Cada gráfico visualiza dos lineas (eje x, eje y ) y (eje x , eje y secundario). 
    Los valores del eje y secundario se obtienen del i-esimo elemento de la lista de nombres de columnas del dataframe.  
        
    Parametros:
    df: dataframe con index temporal ejemplo: date_time
    var_y: string, nombre de variable del eje y
    list: lista de nombres de columnas del eje y secundario, coinciden con la cantidad de gráficas
    alto: alto de la figura que contiene los subgráficas
    largo: largo de la figura que contiene los subgráficas
 '''''
 fig, axes= plt.subplots(nrows=len(list), figsize=(largo,alto))
 axe = axes.ravel()
 for i in range(0,len(list)):
   # ax = fig.add_subplot(111)
   axe[i].plot(df.index,df[var_y], color=colors[0], label = var_y)

   ax2 = axe[i].twinx()
   ax2.plot(df.index,df[list[i]],  color=colors[1], label =list[i])
   axe[i].set_ylabel(r" "+var_y)
   ax2.set_ylabel(r" "+list[i])
   ax2.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axe[i].transAxes)
 return plt.show()

def zoom_lineplot(zoom,data, col):
   '''''
    Devuelve una gráfica con zoom en un rango de fecha dado por el parámetro zoom.   
        
    Parametros:
    data: dataframe cuyo index son valores temporales Ejemplo: date_time
    zoom: rango de fecha ej: ('2021-06-21 14:00:00','2021-06-28 14:00:00')
    col: string, nombre de variable a visualizar.
  '''''
   fig = plt.figure(figsize=(12, 6))
   grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

   main_ax = fig.add_subplot(grid[1:3, :])
   zoom_ax = fig.add_subplot(grid[5:, :])

   data[col].plot(ax=main_ax, alpha=0.5, linewidth=0.5)
   min_y = min(data[col])
   max_y = max(data[col])
   main_ax.fill_between(zoom, min_y, max_y, alpha=0.5, zorder=0)
   main_ax.set_xlabel('')

   data[col].loc[zoom[0]: zoom[1]].plot(ax=zoom_ax,  linewidth=2)

   main_ax.set_title( f'{col}:{data.index.min()}, {data.index.max()}', fontsize=14)
   zoom_ax.set_title( f'{col}: {zoom}', fontsize=14)
   plt.subplots_adjust(hspace=1)
   return plt.show()


def simple_sns_lineplot_marker(df,var_x,var_y,y_marker,title):
 '''''
    Devuelve un gráfico de linea, con una recta roja que atraviesa el eje y, en el punto dado por y_marker.
        
    Parametros:
    var_x: vector, columna del dataframe del eje x
    var_y: vector, columna del dataframe del eje y
    y_marker: valor del eje y donde se desea poner la marca
    title: texto con el título del gráfico
 '''''
 plt.figure(figsize = (20,3))
 sns.lineplot(data=df, x = var_x, y= var_y )
 plt.axhline(y_marker, color = 'r', ls = '--' )
 plt.title(title)
 return plt.show()


def multiple_sns_lineplot_secundary_y_axis(df,var_x, var_y,list,alto,largo):
 '''''
    Devuelve varios i-gráficos según la cantidad de elementos en list. Cada gráfico visualiza dos lineas (eje x, eje y ) y (eje x , eje y secundario). 
    Los valores del eje y secundario se obtienen del i-esimo elemento de la lista de nombres de columnas del dataframe.  
        
    Parametros:
    df: dataframe
    var_x: string, nombre de variable del eje x
    var_y: string, nombre de variable del eje y
    list: lista de nombres de columnas del eje y secundario, coinciden con la cantidad de gráficas
    alto: alto de la figura que contiene los subgráficas
    largo: largo de la figura que contiene los subgráficas
 '''''
 fig, ax = plt.subplots(nrows=len(list), figsize=(largo,alto))
 for i in range(0,len(list)):
  sns.lineplot(data = df, x =var_x, y=var_y,ax=ax[i], color=colors[0])
  sns.lineplot(data = df, x =var_x, y=list[i],ax=ax[i].twinx(),color=colors[1])
  ax[i].set_title(var_y+' vs '+list[i])

 return plt.show()


def multiple_sns_lineplot(df,var_x, list,alto,largo):
 '''''
    Devuelve varios i-gráficos según la cantidad de elementos en list. Cada gráfico visualiza una linea a través del eje x, eje y
    Los valores del eje y se obtienen del i-esimo elemento de la lista de nombres de columnas del dataframe.  
        
    Parametros:
    df: dataframe
    var_x: string, nombre de variable del eje x
    list: lista de nombres de columnas del dataframe a vizualizar en el eje y, coinciden con la cantidad de gráficas
    alto: alto de la figura que contiene los subgráficas
    largo: largo de la figura que contiene los subgráficas
 '''''
 fig, ax = plt.subplots(nrows=len(list), figsize=(largo,alto))
 for i in range(0,len(list)):
     sns.lineplot(data = df, x =var_x, y=list[i], ax=ax[i]) 
    
 return plt.show()

def simple_sns_lineplot_two_y_axis(df,var_x,var_y_axis,var_secundary_axis,largo,alto):
 '''''
    Devuelve una gráfica con un eje y, además de otro eje y secundario.   
        
    Parametros:
    df: dataframe
    var_x: string, nombre de variable del eje x
    var_y_axis: string, nombre de variable del eje y
    var_secundary_axis: string, nombre de variable del eje y secundario
    alto: alto de la figura que contiene los subgráficas
    largo: largo de la figura que contiene los subgráficas 
 '''''
 plt.figure(figsize=(largo,alto))
 sns.lineplot(data = df, x =var_x, y=var_y_axis,label=var_y_axis,color = colors[0])
 sns.lineplot(data = df, x =var_x, y=var_secundary_axis,ax=plt.twinx(),label=var_secundary_axis, color = colors[1])
 plt.legend()

 return  plt.show()

def simple_sns_multilineplot(df,var_x,list,largo,alto,ylabel):
 '''''
    Devuelve una gráfica multi-linea.   
        
    Parametros:
    df: dataframe
    var_x: string, nombre de variable del eje x
    list: lista de nombres de columnas con el mismo rango de valores del eje y
    alto: alto de la figura que contiene los subgráficas
    largo: largo de la figura que contiene los subgráficas 
    ylabel: nombre del eje y
 '''''
 plt.figure(figsize=(largo,alto))
 for i in range(0,len(list)):
   sns.lineplot(data = df, x =var_x, y=list[i], label=list[i])

 plt.ylabel(ylabel)
 plt.legend()
 return plt.show()


def make_patch_spines_invisible(ax):
 '''''
 Función auxiliar de lineplot_three_y_axis
 
 '''''

 ax.set_frame_on(True)
 ax.patch.set_visible(False)
 for sp in ax.spines.values():
    sp.set_visible(False)



def simple_lineplot_three_y_axis(serie1,serie2,serie3,label1, label2, label3,labelx):
   '''''
    Recibe tres series con un mismo index, y devuelve un gráfico con tres ejes y(ejes verticales), con escalas diferentes
        
    Parametros:
    serie1: serie con el mismo index que serie 2 y 3
    serie2: serie con el mismo index que serie 1 y 3
    serie3: serie con el mismo index que serie 1 y 2
    label1: etiqueta del eje y perteneciente a la serie 1
    label2: etiqueta del eje y perteneciente a la serie 2
    label3: etiqueta del eje y perteneciente a la serie 3
    labelx: etiqueta del eje x perteneciente al index de las tres series.

    ejemplo: 
    serie1=df.groupby("Fecha")['OUT_PID'].mean()
    serie2=df.groupby("Fecha")['Flujo_OUT'].mean()
    serie3=df.groupby("Fecha")['Presion'].mean()
    
    Todas las series tienen el mismo index ya que fueron dataframes agrupados por la misma columna (Fecha en este caso). 

    '''''
   fig, host = plt.subplots(figsize=(20,5))
   fig.subplots_adjust(right=0.75)

   par1 = host.twinx()
   par2 = host.twinx()

   # Offset the right spine of par2.  The ticks and label have already been
   # placed on the right by twinx above.
   par2.spines["right"].set_position(("axes", 1.2))
   # Having been created by twinx, par2 has its frame off, so the line of its
   # detached spine is invisible.  First, activate the frame but make the patch
   # and spines invisible.
   make_patch_spines_invisible(par2)
   # Second, show the right spine.
   par2.spines["right"].set_visible(True)

   p1, = host.plot(serie1.index, serie1, color = colors[0], label=label1)
   p2, = par1.plot(serie2.index, serie2, color = colors[1], label=label2)
   p3, = par2.plot(serie3.index, serie3, color = colors[2], label=label3)

   lns = [p1, p2, p3]
   
   host.set_xlabel(labelx)
   host.set_ylabel(label1)
   par1.set_ylabel(label2)
   par2.set_ylabel(label3)

   host.yaxis.label.set_color(p1.get_color())
   par1.yaxis.label.set_color(p2.get_color())
   par2.yaxis.label.set_color(p3.get_color())

   tkw = dict(size=4, width=1.5)
   host.tick_params(axis='y', colors=p1.get_color(), **tkw)
   par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
   par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
   host.tick_params(axis='x', **tkw)
  
   lines = [p1, p2, p3]
   host.legend(lines, [l.get_label() for l in lines])
   return plt.show()

##### Correlations functions #####################################

def find_correlated_features(df, threshold, target_variable, down=False):
 '''''
    Devuelve una lista de variables correlacionadas con la target_variable, cuya correlación sea por encima de un umbral.
        
    Parametros:
    df: dataframe
    threshold: valor que representa el umbral
    target_variable: string, nombre de la variable a correlacionar  
 

    '''''
 s = df.corr().loc[target_variable].drop(target_variable)
 if down:  return s[s.abs() <= threshold]
 else: return s[s.abs() >= threshold]

def corr_matrix_pearson(df, dim):
  ''''
    Devuelve una lista de variables correlacionadas con la target_variable, cuya correlación sea por encima de un umbral.
        
    Parametros:
    df: dataframe
    threshold: valor que representa el umbral
    target_variable: string, nombre de la variable a correlacionar  
 

    '''''

  corr_matrix = df.corr(method='pearson')
  corr_matrix

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(dim, dim))

  sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 8},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
  )

  ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
  )

  ax.tick_params(labelsize = 10)
  return plt.show()

############################# scatterplots ###############################################

def sns_joinplot(df, x, y,hue_value='', hue=False ):
 '''''
    Devuelve un scatterplot con histogramas de frecuencia en los márgenes. Adicionalmente, 
    si la variable hue es TRue, se separan por colores las observaciones,
    y los histogramas se transforman en curvas de densidad separadas pr colores.
        
    Parametros:
    df: dataframe con el set de datos. Debe tener en principio las columnas x, y
    x: String, nombre de variable del eje x.
    y: String, nombre de variable del eje y. 
    hue_value:String, nombre de variable categórica por la que se van a separar las observaciones,
                      en caso que la variable hue sea True. Por defecto es ''
    hue: Logical. 
 '''''


 if hue: sns.jointplot(data=df, x=x, y=y,hue=hue_value,marginal_ticks=True)
 else: sns.jointplot(data=df, x=x, y=y, marginal_ticks=True)

 return plt.show()



def sns_joinplot_hex(df, x, y):
 '''''
    Devuelve un scatterplot con hexágonos de densidad en las observaciones, 
    e histogramas de frecuencia en los márgenes.
        
    Parametros:
    df: dataframe con el set de datos. Debe tener en principio las columnas x, y
    x: String, nombre de variable del eje x.
    y: String, nombre de variable del eje y.  
    '''''

 sns.jointplot(data=df, x=x, y=y,kind='hex', marginal_ticks=True)
 return plt.show()


def sns_pairplot(df, col_hue):
  '''''
    Devuelve una matriz de scatterplots del dataframe df. Esta gráfica separa las observaciones 
    por colores según una variable categórica col_hue. 
    Esta función es costosa en cuanto a tiempo de ejecución. 
    Parametros:
    df: dataframe
    col_hue: string, nombre de la columna de categorías para separar las observaciones por colores. 
    

    '''''

  sns.pairplot(df, hue=col_hue)
  return plt.show()

def multiple_sns_scatterplot_one_vs_all(df,features,var,fig_largo,fig_alto):
 '''''
    Devuelve multiples scatterplots de la variable var respecto a las variables almacenadas en features. Esta función recibe en features
    un listado de las variables a graficar, estas pueden obtenidas mediante la función "find_correlated_features(df, threshold, target_variable)"
        
    Parametros:
    df: dataframe
    features: variables a o nombres de columnas en el eje y
    var: string, nombre de la variable eje x
    fig_alto: alto de la figura que contiene los subgráficas
    fig_largo: largo de la figura que contiene los subgráficas 

    '''''

 if len(features)<=3:  fig, axes = plt.subplots(ncols=len(features), nrows=1, figsize=(fig_largo, fig_alto))
 else : fig, axes = plt.subplots(ncols=3, nrows=int(len(features)/3+1), figsize=(fig_largo, fig_alto)) 
 
 plotted = {}
 axes = axes.flatten() 
 for i in range(0,len(axes)):
    plotted[i] = 0
    if i<len(features) :
      sns.scatterplot(x=var,y=features[i], data=df, ax=axes[i])
      plotted[i] = 1


 for plot,ax in zip(plotted,axes):
  if plotted[plot] == 0:   
    ax.remove()
 return plt.show()



#########  heatmap ##############################

def simple_july_heatmap(myserie, title):
   '''''
    Devuelve un heatmap de la serie pasada por parámetro. El parámetro serie recibe una serie 
    de valores enteros cuyo index debe ser un rango de fechas.
    
   Fecha
   2021-10-02    12
   2021-10-03     0
   2021-10-04     1
   .......
   2021-10-31     0
   2021-11-01     2
   2021-11-02     1
   Name: serie1, dtype: int64 

    Parametros:
    serie: serie de enteros con index rango de fechas. 
    title: titulo del gráfico.


    '''''
   july.heatmap(myserie.index, myserie, cmap="golden", value_label=True,title=title)
   return plt.show()
    


def multiple_july_heatmap(rows,cols,largo,alto, list_varname,list_title):
  '''''
    Devuelve una grid de heatmap, cada celda es un heatmap de una serie cuyo index debe ser un rango de fechas.  
           
    Parametros:
    rows:cantidad de filas de la grid
    cols: cantidad de columnas de la grid
    largo: largo de la grid
    alto: alto de la grid
    list_varname: lista de series a visualizar. 
    list_title: lista de títulos de los heatmaps

Ejemplo de serie que recibe list_varname:  serie1 es una serie con index Fecha, 
cuyos valores son la cantidad de lecturas con valores null en ese día.  
Fecha
2021-10-02    12
2021-10-03     0
2021-10-04     1
.......
2021-10-31     0
2021-11-01     2
2021-11-02     1
Name: serie1, dtype: int64 
    '''''
 
  fig, axes = plt.subplots(rows,cols, figsize=(largo,alto))
  for i in range(0,len(list_varname)):
    july.heatmap(list_varname[i].index, list_varname[i],  value_label=True,title=list_title[i], ax=axes[i])
  return plt.show()

##################### Función específica ###################################

def plot_var(df, xvar, yvar, y2var = False, range_date = False, cycle_id = False, show_cycle = True, figsize=(20,3), kind = 'line', hline = False):
  '''
  Funcion específica para dataframes que contengan las columnas cycle_id y date_time. 

  plot_var: permite estudiar variables de serie de tiempo, en este grafico la variable tiempo se muestra en el eje x, pudiendo comparar dos variables 
  en el eje y. Se puede graficar por rango de fecha considerando como columna ['date_time'] o graficar por un ciclo en particular, para hacer mejor uso
  de esta funcion es necesario utilizar el dataframe luego del ETL. 
  Parametros
  - df: dataframe que contiene los datos a graficar, debe contener una variable de fecha y tiempo 'date_time'
  - xvar: string, nombre de la variable tiempo que se quiere mostrar en el eje x, por ejemplo 'date_time'
  - yvar: string, nombre de la variable a analizar respecto al tiempo, ej: 'do_level'
  - y2var: string, nombre de la segunda variable que se quiere analizar en simultaneo, esta se mostrará en el eje y secundario. En caso de no utilizar la segunda variable
          se mantiene como False por defecto
  - range_date: rango de fechas que se quieren visualizar, debe ser ingresada como lista con los valores tipo string, por ejemplo ['2020-05-12','2020-05-18'], 
                False por defecto, en caso de querer visualizar un solo ciclo se debe mantener esta variable como False
  - cycle_id: tipo int, corresponde al id del ciclo que se quiere graficar, False por defecto, en caso de querer graficar por cycle id se debe
              mantener la variable range_date como False
  - show_cycle: variable booleana que muestra los ciclos de la serie de tiempo
  - figsize = tamaño de la figura, por defecto (20,3)
  - kind = tipo de grafica a visualizar, por defecto 'line'
  - hline = grafica una linea horizonta con el valor ingresado, por defecto es False 
  '''
  
  if range_date != False:
    df_tmp = df[(df['date_time'].astype(str) >= range_date[0]) & (df['date_time'].astype(str) <= range_date[1])].copy()
    print(f'Historico de periodo {range_date[0]} - {range_date[1]}')
  elif cycle_id != False:
    df_tmp = df[df.cycle_id == cycle_id].copy()
    print(f'Historico de ciclo {cycle_id}')
  ax = df_tmp.plot( x= xvar, y= yvar, figsize=figsize, kind = kind)
  ax.set_ylabel(yvar)

  if hline != False:
    plt.axhline(hline, ls = '--', color = 'black')
    
  if y2var != False:
    ax2 = df_tmp.plot( x= xvar, y= y2var, secondary_y = True, figsize=figsize, kind = kind, ax=ax)
    ax2.set_ylabel(y2var)
        
  if show_cycle == True:
    cycle_date = df_tmp.groupby(['cycle_id'])['date_time'].min().reset_index()
    cycle_id = cycle_date['cycle_id'].to_list()
    for ciclo in range(len(cycle_date)):
      plt.axvline(cycle_date['date_time'][ciclo], color= 'r', ls='--')
      plt.text( x = cycle_date['date_time'][ciclo], y = 5, s = f'Ciclo {cycle_id[ciclo]}')
  plt.grid(visible = False) 
  plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
  plt.show()

################### Boxplots ##################################

def plot_sns_box_and_dot(df, x_var, y_var, hue_var = None):
  '''
  plot_box_and_dot: entrega una figura de boxplot de la variable en estudio y muestra las observaciones en forma de puntos
  Parametros:
  - df: dataframe 
  - x_var: str, nombre ded la variable en eje x, generalmente categoria 
  - y_var: str, nombre de la variable en eje y
  - hue_var: str, nombre de la variable por la que se quiere categorizar, None por defecto
  '''
  #sns.set_style('white', {'legend.frameon':True})
  sns.boxplot(data = df, x=x_var, y = y_var, hue = hue_var)
  sns.stripplot(data = df, x=x_var, y = y_var, hue = hue_var,  color = 'black' )
  plt.show()

def simple_boxplot(df, category,colname ):
  '''
  box_plot: entrega una figura de boxplot de la variable en estudio y muestra las observaciones en forma de puntos
  Parametros:
  - df: dataframe 
  - x_var: str, nombre ded la variable en eje x, generalmente categoria 
  - y_var: str, nombre de la variable en eje y
  - hue_var: str, nombre de la variable por la que se quiere categorizar, None por defecto
  '''
  fig, ax = plt.subplots(figsize=(10, 3.5))
  df.boxplot(column=colname, by=category, ax=ax)
  #df.groupby(category)[colname].mean().plot(style='o-', linewidth=0.8, ax=ax)
  ax.set_ylabel(colname)
  ax.set_title('Distribución '+ colname + ' por '+ category)
  fig.suptitle('')
  return plt.show()

  