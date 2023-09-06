document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const histogramCanvas = document.getElementById("histogramCanvas");
    const ctx = histogramCanvas.getContext("2d");

    imageInput.addEventListener("change", function (e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = new Image();
                img.src = e.target.result;

                img.onload = function () {
                    histogramCanvas.width = img.width;
                    histogramCanvas.height = img.height;

                    ctx.drawImage(img, 0, 0);

                    const imageData = ctx.getImageData(0, 0, img.width, img.height);
                    const histogram = createHistogram(imageData);

                    plotHistogram(histogram);
                };
            };
            reader.readAsDataURL(file);
        }
    });

    function createHistogram(imageData) {
        const histogram = new Array(256).fill(0);

        for (let i = 0; i < imageData.data.length; i += 4) {
            const brightness = Math.floor(
                (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3
            );
            histogram[brightness]++;
        }

        return histogram;
    }

    function plotHistogram(histogram) {
        const histogramCanvas = document.getElementById("histogramCanvas");
        const ctx = histogramCanvas.getContext("2d");
        ctx.clearRect(0, 0, histogramCanvas.width, histogramCanvas.height);

        const maxFrequency = Math.max(...histogram);

        for (let i = 0; i < histogram.length; i++) {
            const barHeight = (histogram[i] / maxFrequency) * histogramCanvas.height;
            ctx.fillStyle = "blue";
            ctx.fillRect(i, histogramCanvas.height - barHeight, 1, barHeight);
        }
    }
});
