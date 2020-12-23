$(document).ready(function () {
    $('#summarize_btn').click(function () {

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarizeNLTK'
        }).done(function (data) {
            $('#summarization').val(data.summarization);
        })

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarizeBERT'
        }).done(function (data) {
            $('#summarization-bert').val(data.summarization);
        })

        console.log('hi');
    });
});
