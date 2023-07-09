from pylatex import Document, Section, Subsection, Tabular, Figure
from datetime import datetime
from matplotlib import pyplot as plt
plt.switch_backend('TkAgg')


def make_report(train_loss_list, val_loss_list, arg_n_values):
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

    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plot_image_path = '../../reports/plot.png'
    plt.savefig(plot_image_path)

    with doc.create(Subsection('Graph: ')):
        with doc.create(Figure(position='h!')) as graph_image:
            graph_image.add_image('plot.png', width='200px')
    doc.generate_pdf('../../reports/report_' + current_time)

    plt.close()
