import streamlit as st

class Page:
    def __init__(self,home_index=0):
        self.pages = []
        self.home = home_index

    def add_page(self, title, func):
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        page = st.sidebar.radio(
            'Menu',
            index=self.home,
            options=self.pages,
            format_func=lambda page: page['title'])

        page['function']()
        page['index'] = 2