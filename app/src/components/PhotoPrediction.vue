<template>
  <div class="photo-prediction">
    <h2 class="title">Prediction:</h2>
    <div class="card-container">
      <el-card
        class="prediction-card card-shadow"
        v-for="([model, age], index) in ageList"
        :key="model"
      >
        <div :class="getAgeClass(index)">{{ age | errorOrRoundInt }}</div>
        <div v-if="showModelName" class="model">- {{ model | modelName }}</div>
      </el-card>
    </div>
  </div>
</template>

<script>
import * as R from 'ramda'

const errorOrRoundInt = R.ifElse(
  R.lt(0),
  R.compose(R.apply(Math.round, R.__), R.of),
  R.always('Err')
)

export default {
  name: 'PhotoPrediction',
  props: {
    ages: {
      type: Object,
      validator: R.compose(
        R.all(R.is(Number)),
        R.props(['cnn', 'regression', 'clustering'])
      )
    },
    showModelName: {
      type: Boolean,
      default: true
    }
  },
  computed: {
    ageList() {
      return R.toPairs(this.ages)
    }
  },
  methods: {
    getAgeClass(index) {
      return R.when(
        () => R.equals(false, this.showModelName),
        R.concat(R.__, ' number-only')
      )(`age color-${index + 1}`)
    }
  },
  filters: {
    modelName: model => {
      if (model === 'cnn') {
        return 'Convolutional Neural Network'
      }
      // TODO: add prettified name for other models
      return model
    },
    errorOrRoundInt
  }
}
</script>

<style scoped lang="scss">
.card-container {
  display: flex;
  justify-content: center;
}

.prediction-card {
  flex: 1;
  box-sizing: border-box;
  min-height: 300px;
  max-width: 320px;
  margin-left: 3rem;
  margin-right: 3rem;
  padding: 2rem;
}

.age {
  padding-top: 0.5rem;
  font-size: 6rem;
  font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
}

.model {
  padding-top: 1.2rem;
  font-style: italic;
  text-transform: capitalize;
  color: var(--desc);
}

.color-1 {
  color: var(--theme2);
}

.color-3 {
  color: var(--theme3);
}

.color-2 {
  color: var(--theme1);
}

.number-only {
  padding-top: 1.2rem;
  font-size: 8rem;
}
</style>
