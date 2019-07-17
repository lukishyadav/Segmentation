#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:58:41 2019

@author: lukishyadav
"""


"""


My Own Interactive plot using Bokeh  



"""



from bokeh.io import curdoc
from bokeh.io import output_file, show
from bokeh.layouts import column,layout,row
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput
from bokeh.plotting import figure

import bokeh.models  as bm

#output_file("layout_widgets.html")

x = list(range(11))
y0 = x
y1 = [10 - i for i in x]
y2 = [abs(i - 5) for i in x]


source=bm.ColumnDataSource(data=dict(x=x, y=y0))


def my_slider_handler(attr,old,new):
    X=source.data['x']
    X[0]=new
    print('Done')
    source.data=(dict(x=X,y=y0))
    #   source.change.emit()


# create some widgets
slider = Slider(start=0, end=10, value=1, step=.1, title="Slider")
slider.on_change("value", my_slider_handler)


button_group = RadioButtonGroup(labels=["Option 1", "Option 2", "Option 3"], active=0)
select = Select(title="Option:", value="foo", options=["foo", "bar", "baz", "quux"])
button_1 = Button(label="Button 1")
button_2 = Button(label="Button 2")


# create a new plot
s1 = figure(plot_width=250, plot_height=250, title=None)
s1.circle('x', 'y',source=source, size=10, color="navy", alpha=0.5)


def my_text_input_handler(attr, old, new):
    print("Previous label: " + old)
    print("Updated label: " + new)

text_input = TextInput(value="default", title="Label:")
text_input.on_change("value", my_text_input_handler)


#show(layout([[s1],[button_1, slider],[button_group, select, button_2],], sizing_mode='stretch_both'))



inputs=column(s1,button_1, slider, button_group, select, button_2,text_input, width=300)

curdoc().add_root(row(inputs,width=800))
curdoc().title = "Interactive"


