from pylatex import Document, Section, Subsection, Tabular


def make_report(info_dict=None):
    doc = Document()
    with doc.create(Section('Report')):
        with doc.create(Subsection('YAML params:')):
            with doc.create(Tabular('l')) as table:

                yaml_file = open('../../config.yaml', 'r')
                lines = yaml_file.readlines()
                table.add_hline()
                for line in lines:
                    table.add_row([line], strict=False)
                table.add_hline()
    doc.generate_pdf('../../models/output/reports/report', clean_tex=False)