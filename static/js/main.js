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

<<<<<<< HEAD
        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarizeLSTM'
        }).done(function (data) {
            $('#summarization-lstm').val(data.summarization);
        })

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarizeBiLSTM'
        }).done(function (data) {
            $('#summarization-bilstm').val(data.summarization);
        })

=======
>>>>>>> 9c466ff724c5d5c6160a85e7b0159b49439043d9
        console.log('hi');
    });
});
