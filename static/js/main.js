$(document).ready(function () {
    $('#summarize_btn').click(function () {

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarize'
        }).done(function (data) {
            $('#summarization').val(data.summarization);
        })
    });
});
