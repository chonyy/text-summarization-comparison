$(document).ready(function () {
    var inputTextLength = $('#input-text').val().length;

    $('#input-length').text(inputTextLength);

    $('#summarize_btn').click(function () {

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarize_NLTK_LSTM_BiLSTM'
        }).done(function (data) {
            $('#summary').val(data.summaryNLTK);
            $('#summary-lstm').val(data.summaryLSTM);
            $('#summary-bilstm').val(data.summaryBiLSTM);

            $('#NLTK-length').text(data.summaryNLTK.length);   
            $('#LSTM-length').text(data.summaryLSTM.length);   
            $('#BiLSTM-length').text(data.summaryBiLSTM.length);   

            $('#nltk-time').text(data.timeNLTK);   
            $('#lstm-time').text(data.timeLSTM);   
            $('#bilstm-time').text(data.timeBiLSTM);   
        })

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
            url: '/summarizeBERT'
        }).done(function (data) {
<<<<<<< HEAD
            $('#summary-bert').val(data.summary);
            $('#BERT-length').text(data.summary.length);
            $('#bert-time').text(data.time);
        })

=======
            $('#summarization-bert').val(data.summarization);
        })

<<<<<<< HEAD
>>>>>>> 8021aad4cd0b850ca61eb43e2c8ffcdedaa5a3d3
        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
<<<<<<< HEAD
            url: '/summarizeT5'
        }).done(function (data) {
            $('#summary-t5').val(data.summary);
            $('#T5-length').text(data.summary.length);
            $('#t5-time').text(data.time);
=======
            url: '/summarizeLSTM'
        }).done(function (data) {
            $('#summarization-lstm').val(data.summarization);
>>>>>>> 8021aad4cd0b850ca61eb43e2c8ffcdedaa5a3d3
        })

        $.ajax({
            data: {
                text: $('#input-text').val()
            },
            type: 'POST',
<<<<<<< HEAD
            url: '/summarizeCNN'
        }).done(function (data) {
            $('#summary-cnn').val(data.summary);
            $('#CNN-length').text(data.summary.length);
            $('#cnn-time').text(data.time);
        })

=======
            url: '/summarizeBiLSTM'
        }).done(function (data) {
            $('#summarization-bilstm').val(data.summarization);
        })

=======
>>>>>>> 9c466ff724c5d5c6160a85e7b0159b49439043d9
>>>>>>> 8021aad4cd0b850ca61eb43e2c8ffcdedaa5a3d3
        console.log('hi');
    });
});
