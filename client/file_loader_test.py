# Write test for file_loader.py

import unittest
from file_loader import create_library, process_text
from langchain.schema import Document

# create Library test case
class CreateLibraryTestCase(unittest.TestCase):
    
    

    def test_main_type_create_library(self):
        output = create_library()
        keys = list(output.keys())
        print(keys)
        print(output)
        self.assertEqual(type(output), dict)

    def test_Document_type_create_library(self):
        self.assertEqual(type(self.output[self.keys[0]]), Document)

class ProcessTextTestCase(unittest.TestCase):
    def test_main_type_process_text(self):
        output = process_text()
        self.assertEqual(type(output), dict)

    def test_Document_type_process_text(self):
        self.assertEqual(type(self.output[self.keys[0]]), Document)