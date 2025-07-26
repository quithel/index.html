# index.html
<!DOCTYPE html>
<html>
<head>
  <title>Red Neuronal Online - Iris</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
</head>
<body>
  <h2>Entrenando Red Neuronal en Tiempo Real...</h2>
  <pre id="log"></pre>

  <script>
    const DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSwrw8_nTV7TZ-k4vmgIJs6-mBONZNxHo16GCFr42Nv-mxUVmGpFq56TmxoeWG2lIw57D1gEBUrOFyM/pubhtml";

    async function fetchCSVData(url) {
      const res = await fetch(url);
      const text = await res.text();
      const rows = text.trim().split("\n").slice(1); // Quitar encabezado
      const data = rows.map(row => {
        const [s1, s2, p1, p2, label] = row.split(",").map(Number);
        const y = [0, 0, 0];
        y[label] = 1;
        return { xs: [s1, s2, p1, p2], ys: y };
      });
      return data;
    }

    async function trainModel(data) {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [4] }));
      model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
      model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      const xs = tf.tensor2d(data.map(d => d.xs));
      const ys = tf.tensor2d(data.map(d => d.ys));

      await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 10,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            document.getElementById('log').textContent += `Época ${epoch + 1}: Precisión ${logs.acc.toFixed(4)}\n`;
          }
        }
      });

      // Guardar modelo local
      await model.save('localstorage://modelo-iris');
      console.log("Modelo guardado.");
    }

    async function autoTrainLoop() {
      while (true) {
        document.getElementById('log').textContent += `\nEntrenando con datos nuevos... 🧠\n`;
        const data = await fetchCSVData(DATA_URL);
        await trainModel(data);
        await new Promise(r => setTimeout(r, 60000)); // Esperar 1 minuto antes de repetir
      }
    }

    autoTrainLoop();
  </script>
</body>
</html>
