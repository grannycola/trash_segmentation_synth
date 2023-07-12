from pylatex import Document, Section, Subsection, Tabular, Figure
from datetime import datetime
from matplotlib import pyplot as plt
import os

plt.switch_backend('TkAgg')


def make_report(train_loss_list,
                val_loss_list,
                train_metric_list,
                val_metric_list,
                arg_n_values):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)

    now = datetime.now()
    current_time = now.strftime("%d.%m.%y_%H-%M-%S")

    with doc.create(Section('Report')):
        with doc.create(Subsection('YAML params:')):
            with doc.create(Tabular('l')) as table:
                table.add_hline()
                for arg, val in arg_n_values:
                    table.add_row([f"{arg}: {val}"], strict=False)
                table.add_hline()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Loss')
    ax1.plot(train_loss_list)
    ax1.plot(val_loss_list)

    ax2.set_title('Metric')
    ax2.plot(train_metric_list)
    ax2.plot(val_metric_list)

    max_index = val_metric_list.index(max(val_metric_list))
    min_index = val_loss_list.index(min(val_loss_list))

    ax1.scatter(min_index, val_loss_list[min_index])
    ax1.annotate(f'Min: {round(val_loss_list[min_index], 3)}',
                 (min_index, val_loss_list[min_index]),
                 textcoords='offset points', xytext=(1, 1))

    ax2.scatter(max_index, val_metric_list[max_index])
    ax2.annotate(f'Max: {round(val_metric_list[max_index], 3)}',
                 (max_index, val_metric_list[max_index]),
                 textcoords='offset points', xytext=(1, 1))

    plot_image_path = '../../reports/plot.png'
    fig.savefig(plot_image_path)

    with doc.create(Subsection('Graph: ')):
        with doc.create(Figure(position='h!')) as graph_image:
            graph_image.add_image('plot.png')
            os.remove(plot_image_path)
    doc.generate_pdf('../../reports/report_' + current_time)
