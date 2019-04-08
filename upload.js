const url = 'process.php';
const form = document.querySelector('form');
const def = "<h2 style='text-align: center;'>Your Results</h2><img id='loading' src='loading.gif' />"

form.addEventListener('submit', e => {
    e.preventDefault();
    
    document.getElementById("result").innerHTML = def;


    if (document.getElementById('imageSel').value === '') {
        window.alert("select an image");
        return
    }

    const files = document.querySelector('[type=file]').files;
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        let file = files[i];

        formData.append('files[]', file);
    }

    document.getElementById('result').style.display = 'block';

    fetch(url, {
        method: 'POST',
        body: formData
    }).then(function(myJson) {
        myJson = myJson.text()
        console.log(myJson);
        return myJson
      }).then(function(data) {
        Promise.resolve(data).then(function(result) {
            console.log(result)

            document.getElementById("result").innerHTML = result;
            document.getElementById("form").reset()
        })
      });

    //console.log(response.responseText)

});