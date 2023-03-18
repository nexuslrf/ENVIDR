const data = {
  datasets: [
    {
      label: 'Optimal',
      backgroundColor: '#FF8888',
      borderColor: '#FF8888',
      pointRadius: 5,
      pointHoverRadius: 7,
      data: [
        {x: 1.60, y: 31.36},
        {x: 5.81, y: 32.41},
        {x: 18.45, y: 33.24},
        {x: 35.30, y: 33.61},
        {x: 69.00, y: 33.79},
      ]
    },
    {
      label: 'Baseline',
      backgroundColor: '#FFCF4F',
      borderColor: '#FFCF4F',
      pointRadius: 5,
      pointHoverRadius: 7,
      data: [
        {x: 1.60, y: 22.08},
        {x: 5.81, y: 22.77},
        {x: 10.03, y: 23.83},
        {x: 18.45, y: 24.80},
        {x: 26.88, y: 25.94},
        {x: 35.30, y: 27.04},
        {x: 43.73, y: 27.76},
        {x: 52.15, y: 29.07},
        {x: 60.58, y: 31.75},
        {x: 69.00, y: 33.79},
      ]
    },
    {
      label: 'Proposed',
      backgroundColor: '#A0D6FF',
      borderColor: '#A0D6FF',
      pointRadius: 5,
      pointHoverRadius: 7,
      data: [
        {x: 1.60, y: 31.19},
        {x: 5.81, y: 32.33},
        {x: 10.03, y: 32.55},
        {x: 18.45, y: 33.16},
        {x: 26.88, y: 33.16},
        {x: 35.30, y: 33.30},
        {x: 43.73, y: 33.50},
        {x: 52.15, y: 33.55},
        {x: 60.58, y: 33.60},
        {x: 69.00, y: 33.66},
      ]
    }
  ]
};

// 0, 1, 2, 3, 4 --> 0, 1, 3, 5, 9
const opt2else = {
  0: 0,
  1: 1,
  2: 3,
  3: 5,
  4: 9,
}

const config = {
  type: 'scatter',
  data: data,
  options: {
    showLine: true,
    scales: {
      x: {
        title: {
          display: true,
          font: { size: 18 },
          text: 'Size (MB)'
        }
      },
      y: {
        title: {
          display: true,
          font: { size: 18 },
          text: 'PSNR'
        }
      }
    },
    plugins: {
      legend: {
        labels: {
          font: { size: 18 },
        }
      }
    },
    // event listener
    onClick: (evt) => {
      const points = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
      if (points.length > 0) {
        // change the images accordingly.
        var dataset_index = points[0].datasetIndex;
        var index = points[0].index;

        // special handling of the missing optimal image...
        if (dataset_index == 0) index = opt2else[index];
          
        if ([0, 1, 3, 5, 9].includes(index)) {
          var opt_file = `images/compression/optimal${index}.jpg`;
        } else {
          var opt_file = 'images/compression/na.jpg';
        }
        
        // update the images
        document.getElementById('img_optimal').src = opt_file;
        document.getElementById('img_baseline').src = `images/compression/baseline${index}.jpg`;
        document.getElementById('img_proposed').src = `images/compression/proposed${index}.jpg`;

      }
    },
  }
};

const chart = new Chart(
  document.getElementById('chart_compression'),
  config
);

