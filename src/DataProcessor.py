import re
import xlrd
import logging


class DataProcessor:
    """ Takes input data for MSGSP and performes processing """

    DATA_FILE_PATH = "../data/training-Obama-Romney-tweets.xlsx"

    def __init__(self, data_file_path, debug=False):
        self.DATA = {}
        self.DATA_FILE_PATH = data_file_path
        self.DEBUG = debug
        self.workbook = xlrd.open_workbook(self.DATA_FILE_PATH)

    def load_excel_data(self, sheet_name):
        sheet = self.workbook.sheet_by_name(sheet_name)

        data_row = {}

        for i in range(sheet.nrows):
            try:
                tweet_text = sheet.row_values(i)[3]
                tweet_text = re.sub('<[^<]+?>', '', tweet_text) #remove html tags

                if tweet_text == '':
                    continue

                class_label = sheet.row_values(i)[4]

                if isinstance(class_label, basestring) and class_label not in ['0', '-1', '1']:
                    continue

                if int(class_label) == 2:
                    continue

                data_row[tweet_text.lower()] = int(class_label)

            except:
                # logging.error("Excel parse error: sheet *" + sheet_name + "* at row: " + str(i))
                pass

        logging.info('data size:' + str(len(data_row.items())))

        self.DATA['text'] = data_row.keys()
        self.DATA['target'] = data_row.values()
        self.DATA['size'] = len(data_row.items())

        return self.DATA

    @staticmethod
    def print_report(report):

        if not report:
            print "None"
            return

        print '%-14s %10s %10s %10s %10s' % ('CLASSIFIER', 'PRECISION', 'RECALL', 'FSCORE', 'ACCURACY')

        for cls, rep in report.iteritems():
            print ''
            print "%-14s %10.2f %10.2f %10.2f %10.2f" % (cls, (100 * report[cls]['precision'][0]), (100 * report[cls]['recall'][0]), (100 * report[cls]['fscore'][0]), (100 * report[cls]['accuracy']))
            print "%-14s %10.2f %10.2f %10.2f" % ('', (100 * report[cls]['precision'][1]), (100 * report[cls]['recall'][1]), (100 * report[cls]['fscore'][1]))

