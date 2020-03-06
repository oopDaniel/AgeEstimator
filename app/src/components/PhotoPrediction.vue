<template>
  <div class="photo-prediction">
    <h2 class="title">Prediction:</h2>
    <div class="card-container">
      <el-card
        class="prediction-card card-shadow"
        v-for="([model, age], index) in ageList"
        :key="model"
      >
        <div :class="getAgeClass(index)">
          <font-awesome-icon
            v-show="hasError(age)"
            class="large-font"
            :icon="['fas', 'bug']"
          />
          <span
            :class="{
              'large-font': !hasError(age),
              'small-font': hasError(age)
            }"
          >
            {{ age | errorOrDisplay }}
          </span>
        </div>
        <div v-if="showModelName" class="model">- {{ model | modelName }}</div>
      </el-card>
    </div>
  </div>
</template>

<script>
import * as R from 'ramda'

const hasError = R.either(R.equals('0'), R.equals('-1'))
const errorOrDisplay = R.when(hasError, R.always('Something went wrong ;('))

export default {
  name: 'PhotoPrediction',
  props: {
    ages: {
      type: Object,
      validator: R.compose(
        R.all(R.is(String)),
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
    hasError,
    getAgeClass(index) {
      return R.when(
        () => R.equals(false, this.showModelName),
        R.concat(R.__, ' largest-font')
      )(`age-box color-${index + 1}`)
    }
  },
  filters: {
    modelName: model => {
      if (model === 'cnn') {
        return 'Convolutional Neural Network'
      }
      if (model === 'regression') {
        return 'Logistic Regression'
      }
      // TODO: add prettified name for clustering
      return model
    },
    errorOrDisplay
  }
}
</script>

<style scoped lang="scss">
.card-container {
  display: flex;
  justify-content: center;
  cursor: pointer;
}

.prediction-card {
  flex: 1;
  box-sizing: border-box;
  min-height: 300px;
  max-width: 320px;
  margin-left: 2rem;
  margin-right: 2rem;
  padding: 3rem 1rem 2rem;
}

.age-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 0.5rem;
  font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
}
.large-font {
  font-size: 5rem;
}
.small-font {
  display: flex;
  align-items: flex-end;
  min-height: 40px;
  color: var(--desc);
}

.largest-font {
  padding-top: 1.2rem;
  &.large-font {
    font-size: 8rem;
  }
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
</style>
