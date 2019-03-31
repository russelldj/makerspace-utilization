/*
    The main script. Mostly just sets the date selection drop down
*/

function init_image()
{
    var date_dropdown = document.getElementById("id_date_select");
    var first_img = date_dropdown.options[0].value;
    date_dropdown.selectedIndex = 0;

    var display = document.getElementById("id_display_img_1");
    display.src = "https://people.clarkson.edu/~gwinndr/HeatMaps/" + first_img + "/heatmap.png";
}

function dropdown_selection()
{
    var display = document.getElementById("id_display_img_1");
    var date_dropdown = document.getElementById("id_date_select");

    var selected_img = date_dropdown.options[date_dropdown.selectedIndex].value
    display.src = "https://people.clarkson.edu/~gwinndr/HeatMaps/" + selected_img + "/heatmap.png";
}
