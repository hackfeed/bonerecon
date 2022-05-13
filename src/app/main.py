import dearpygui.dearpygui as dpg


def saveData(dest, src):
    dest = src


def main():
    filepath = ""

    dpg.create_context()
    dpg.create_viewport(title="bonerecon", width=600, height=400, resizable=False)
    dpg.setup_dearpygui()

    with dpg.font_registry():
        with dpg.font("/Users/kononenko/Library/Fonts/consola.ttf", 12) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)

    with dpg.file_dialog(directory_selector=False, show=False, callback=lambda _, x: saveData(filepath, x['file_path_name']), id="file_dialog_id"):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.bind_font(default_font)

    with dpg.window(tag="primary"):
        dpg.add_button(label="Выберите файл", callback=lambda: dpg.show_item("file_dialog_id"))
        dpg.add_checkbox()
        dpg.bind_font(default_font)

    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
