<template>
  <div class="photo-uploader">
    <h1>{{ msg }}</h1>
    <section>
      <div>Upload your image to estimate the age</div>
      <div class="desc">(Less then 2MB. Square image size preferred)</div>
      <section class="card-section center">
        <el-card class="card-box card-shadow">
          <el-upload
            :class="{
              'avatar-uploader': true,
              'post-upload': hasUploaded
            }"
            action="/api/predict_age"
            :show-file-list="false"
            :on-success="handleAvatarSuccess"
            :before-upload="beforeAvatarUpload"
            :multiple="false"
          >
            <i class="el-icon-plus avatar-uploader-icon"></i>
          </el-upload>
          <div v-if="imageUrl !== ''" class="preview-box">
            <transition name="fade">
              <img
                width="100%"
                :src="imageUrl"
                alt=""
                @load="onImgLoaded"
                v-show="isImgReady"
              />
            </transition>
          </div>
        </el-card>
      </section>
      <PhotoPrediction
        class="prediction"
        v-if="hasUploaded"
        :ages="prediction.ages"
      />
      <div
        class="upload-hint"
        v-if="isLoading"
        v-loading="true"
        element-loading-text="Predicting..."
      ></div>
      <div class="upload-hint error" v-show="errorMessage !== ''">
        <font-awesome-icon :icon="['fas', 'radiation-alt']" />
        {{ errorMessage }}
      </div>
    </section>
  </div>
</template>

<script>
import * as R from 'ramda'
import PhotoPrediction from './PhotoPrediction'

const VALID_FILE_TYPES = ['image/png', 'image/jpeg']
const UNSUPPORTED_MSG = 'Unsupported picture format. Please upload a JPG or PNG'
const TOO_LARGE_MSG = 'Picture size can not exceed 2MB'
const MAX_SIZE = 1024 ** 2

const isNotNil = R.complement(R.isNil)
const isNotNilByProp = prop => R.both(isNotNil, R.propSatisfies(isNotNil, prop))

export default {
  name: 'PhotoUploader',
  props: {
    msg: {
      type: String,
      default: ''
    }
  },
  data: () => ({
    imageUrl: '',
    activeFileName: '',
    prediction: null,
    isLoading: false,
    isImgReady: false
  }),
  computed: {
    hasUploaded() {
      return isNotNilByProp('ages')(this.prediction)
    },
    errorMessage() {
      return R.compose(
        R.ifElse(
          R.both(isNotNilByProp('message'), R.propSatisfies(R.isNil, 'ages')),
          R.prop('message'),
          R.always('')
        )
      )(this.prediction)
    }
  },
  methods: {
    handleAvatarSuccess(prediction, file) {
      if (prediction.name === this.activeFileName) {
        this.prediction = prediction
        this.imageUrl = URL.createObjectURL(file.raw)
      }
      this.isLoading = false
      this.isImgReady = false
    },
    beforeAvatarUpload({ type, size, name }) {
      this.prediction = null
      this.imageUrl = ''

      const isValidType = VALID_FILE_TYPES.includes(type)
      const isLt2M = size / MAX_SIZE < 2

      if (!isValidType) {
        this.$message.error(UNSUPPORTED_MSG)
      } else if (!isLt2M) {
        this.$message.error(TOO_LARGE_MSG)
      } else {
        this.activeFileName = name
        this.isLoading = true
      }

      return isValidType && isLt2M
    },
    onImgLoaded() {
      this.isImgReady = true
    }
  },
  components: {
    PhotoPrediction
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.photo-uploader {
  color: var(--title);
  padding-bottom: 6rem;
}

.desc {
  margin-top: 0.2rem;
  color: var(--desc);
  font-style: italic;
}

.card-section {
  margin-top: 2rem;
  flex-direction: column;
}
.card-box {
  position: relative;
  max-width: 400px;
  width: 320px;
  height: 320px;
}

.preview-box {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80%;
  max-width: 320px;
  cursor: pointer;
}

.prediction {
  margin-top: 2rem;
}
.upload-hint {
  margin-top: 5rem;
}
.error {
  color: var(--danger);
}

.fade-enter-active {
  transition: opacity 0.5s ease-in-out;
}
.fade-enter-to {
  opacity: 1;
}
.fade-enter {
  opacity: 0;
}
</style>

<style lang="scss">
.photo-uploader {
  .el-card__body {
    position: relative;
    height: 100%;
    padding: 0;
  }
  .avatar-uploader {
    position: absolute;
    top: calc(50% - 2px);
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
  }

  .el-upload {
    border: 1px dashed #d9d9d9;
    border-radius: 6px;
    position: relative;
    cursor: pointer;
    transition: border-color 0.2s;
    &:hover {
      border-color: var(--theme1);
    }
  }

  .post-upload {
    .el-upload {
      border-color: transparent;
      &:hover {
        border-color: var(--theme1);
      }
    }
    .avatar-uploader-icon {
      color: transparent;
      &:hover {
        color: var(--desc);
      }
    }
  }

  .avatar-uploader-icon {
    font-size: 28px;
    color: var(--desc);
    width: 256px;
    height: 256px;
    line-height: 256px;
    text-align: center;
    transition: color 0.2s;
  }
  .avatar {
    width: 256px;
    height: 256px;
    display: block;
  }
}

.card-shadow {
  box-shadow: 0 2px 6px 0 rgba(0, 0, 0, 0.1);
  &:hover {
    box-shadow: 0 8px 20px 0 rgba(0, 0, 0, 0.2);
  }
}
</style>
