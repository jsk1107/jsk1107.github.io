$(function() {
    $(".container").on("click", function() {
        var that = $(this);
        if (that.hasClass("is-open")) {
            that.removeClass("is-open").addClass("is-closed");
        } else {
            that.removeClass("is-closed").addClass("is-open");
        }
    });
});


